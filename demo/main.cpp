#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <base/base.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

#ifndef KLLM_MODEL_HEADER
#define KLLM_MODEL_HEADER "model/llama3.h"
#endif

#include KLLM_MODEL_HEADER

#ifndef KLLM_MODEL_CLASS
#define KLLM_MODEL_CLASS model::LLama2Model
#endif

namespace {

using json = nlohmann::json;

struct GenerationResult {
  std::string response;
  int32_t steps = 0;
  bool cancelled = false;
};

struct ServeState {
  std::atomic<bool> exit_requested{false};
  std::atomic<bool> cancel_requested{false};
  std::condition_variable cv;
  std::mutex mutex;
  std::queue<std::string> prompts;
};

base::TokenizerType detect_tokenizer_type(const char* tokenizer_path) {
  if (tokenizer_path == nullptr) {
    return base::TokenizerType::kEncodeBpe;
  }
  std::string path = tokenizer_path;
  std::transform(path.begin(), path.end(), path.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (path.size() >= 6 && path.substr(path.size() - 6) == ".model") {
    return base::TokenizerType::kEncodeSpe;
  }
  return base::TokenizerType::kEncodeBpe;
}

void configure_rope_theta_env(const char* checkpoint_path, const char* tokenizer_path) {
  namespace fs = std::filesystem;
  std::vector<fs::path> candidates;
  if (checkpoint_path != nullptr && *checkpoint_path != '\0') {
    candidates.emplace_back(fs::path(checkpoint_path).parent_path() / "config.json");
  }
  if (tokenizer_path != nullptr && *tokenizer_path != '\0') {
    const fs::path tokenizer_config = fs::path(tokenizer_path).parent_path() / "config.json";
    if (std::find(candidates.begin(), candidates.end(), tokenizer_config) == candidates.end()) {
      candidates.push_back(tokenizer_config);
    }
  }

  for (const auto& config_path : candidates) {
    if (!fs::exists(config_path)) {
      continue;
    }
    try {
      std::ifstream input(config_path);
      json payload = json::parse(input);
      if (!payload.contains("rope_theta")) {
        continue;
      }
      const double rope_theta = payload.at("rope_theta").get<double>();
      if (rope_theta <= 0.0) {
        continue;
      }
      const std::string rope_text = std::to_string(rope_theta);
      setenv("KLLM_ROPE_THETA", rope_text.c_str(), 1);
      return;
    } catch (const std::exception&) {
      continue;
    }
  }
  unsetenv("KLLM_ROPE_THETA");
}

void emit_load_progress_json(size_t loaded_bytes, size_t total_bytes, const std::string& stage) {
  std::cout << "[LOAD_PROGRESS]"
            << json{{"loaded_bytes", loaded_bytes},
                    {"total_bytes", total_bytes},
                    {"stage", stage}}
                   .dump()
            << std::endl;
  std::cout.flush();
}

void clear_tensor_buffer(const tensor::Tensor& tensor) {
  const auto buffer = tensor.get_buffer();
  if (!buffer || buffer->ptr() == nullptr || buffer->byte_size() == 0) {
    return;
  }
  const auto allocator = buffer->allocator();
  if (!allocator) {
    return;
  }
  allocator->memset_zero(buffer->ptr(), buffer->byte_size(), nullptr, true);
}

void reset_generation_state(const KLLM_MODEL_CLASS& model) {
  using model::ModelBufferType;
  const std::vector<ModelBufferType> runtime_buffers = {
      ModelBufferType::kInputTokens,     ModelBufferType::kInputEmbeddings,
      ModelBufferType::kOutputRMSNorm,   ModelBufferType::kKeyCache,
      ModelBufferType::kValueCache,      ModelBufferType::kQuery,
      ModelBufferType::kInputPos,        ModelBufferType::kScoreStorage,
      ModelBufferType::kOutputMHA,       ModelBufferType::kAttnOutput,
      ModelBufferType::kW1Output,        ModelBufferType::kW2Output,
      ModelBufferType::kW3Output,        ModelBufferType::kFFNRMSNorm,
      ModelBufferType::kForwardOutput,   ModelBufferType::kForwardOutputCPU,
  };

  for (const auto buffer_type : runtime_buffers) {
    if (model.has_buffer(buffer_type)) {
      clear_tensor_buffer(model.get_buffer(buffer_type));
    }
  }
}

std::string trim(const std::string& text) {
  const auto start = text.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(start, end - start + 1);
}

std::string remove_end_markers(const std::string& text) {
  std::string output = text;
  const std::vector<std::string> end_markers = {
      "<|im_end|>", "<|endoftext|>", "<|end|>", "</s>", "<|eot_id|>", "<|end_of_text|>"};
  for (const auto& marker : end_markers) {
    const auto marker_pos = output.find(marker);
    if (marker_pos != std::string::npos) {
      output = output.substr(0, marker_pos);
    }
  }
  return trim(output);
}

GenerationResult generate(const KLLM_MODEL_CLASS& model, const std::string& sentence,
                          int max_new_tokens,
                          const std::atomic<bool>* cancel_requested = nullptr) {
  reset_generation_state(model);
  auto tokens = model.encode(sentence);
  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  if (tokens.empty()) {
    return {"", 0, false};
  }

  const int32_t max_seq_len = model.max_seq_len();
  const int32_t requested_max_new_tokens = std::max(1, max_new_tokens);
  if (max_seq_len > 0 && prompt_len >= max_seq_len) {
    std::ostringstream error;
    error << "当前请求超过模型上下文长度：prompt_tokens=" << prompt_len
          << ", max_token=" << requested_max_new_tokens << ", seq_len=" << max_seq_len
          << "。请减少历史轮数或降低 max_token。";
    return {error.str(), 0, false};
  }
  int32_t effective_max_new_tokens = requested_max_new_tokens;
  if (max_seq_len > 0 && prompt_len + requested_max_new_tokens > max_seq_len) {
    effective_max_new_tokens = std::max(1, max_seq_len - prompt_len);
    std::fprintf(stderr,
                 "[WARN] max_new_tokens clamped from %d to %d because prompt_tokens=%d and "
                 "seq_len=%d\n",
                 requested_max_new_tokens, effective_max_new_tokens, prompt_len, max_seq_len);
  }

  const int32_t total_steps = prompt_len + effective_max_new_tokens;
  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  int32_t last_generated = -1;
  int32_t same_token_run = 0;
  bool cancelled = false;

  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  std::vector<int32_t> words;
  words.push_back(next);

  while (pos < total_steps) {
    if (cancel_requested != nullptr && cancel_requested->load()) {
      cancelled = true;
      break;
    }

    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }

    if (cancel_requested != nullptr && cancel_requested->load()) {
      cancelled = true;
      break;
    }
    if (!is_prompt && model.is_sentence_ending(next)) {
      break;
    }

    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
      if (next == last_generated) {
        same_token_run += 1;
      } else {
        last_generated = next;
        same_token_run = 1;
      }
      if (same_token_run >= 24) {
        break;
      }
    }
    pos += 1;
  }

  std::string response;
  if (words.size() > static_cast<size_t>(prompt_len)) {
    std::vector<int32_t> response_tokens(words.begin() + prompt_len, words.end());
    response = remove_end_markers(model.decode(response_tokens));
  }
  return {response, std::min(pos, total_steps), cancelled};
}

bool init_model(KLLM_MODEL_CLASS& model) {
  model.set_load_progress_callback(
      [](size_t loaded_bytes, size_t total_bytes, const std::string& stage) {
        emit_load_progress_json(loaded_bytes, total_bytes, stage);
      });

  base::DeviceType device_type = base::DeviceType::kDeviceCUDA;
  const char* force_device = std::getenv("KLLM_DEVICE");
  if (force_device != nullptr && std::string(force_device) == "cpu") {
    device_type = base::DeviceType::kDeviceCPU;
  }

  auto init_status = model.init(device_type);
  if (!init_status && device_type == base::DeviceType::kDeviceCUDA) {
    std::fprintf(stderr, "[WARN] CUDA init failed, fallback to CPU.\n");
    init_status = model.init(base::DeviceType::kDeviceCPU);
  }
  if (!init_status) {
    std::fprintf(stderr, "[ERROR] model init failed, code=%d\n", init_status.get_err_code());
    return false;
  }
  return true;
}

int parse_max_steps(const char* raw, int default_steps = 128) {
  if (raw == nullptr) {
    return default_steps;
  }
  int value = std::atoi(raw);
  if (value <= 0) {
    return default_steps;
  }
  return value;
}

float parse_temperature(const char* raw, float default_temperature = 0.0f) {
  if (raw == nullptr) {
    return default_temperature;
  }
  float value = std::atof(raw);
  if (value < 0.0f) {
    return default_temperature;
  }
  return value;
}

void print_result(const GenerationResult& result, double duration_sec, bool print_stats) {
  const double steps_per_s =
      duration_sec > 0 ? static_cast<double>(result.steps) / duration_sec : 0.0;
  std::printf("[RESPONSE_START]\n%s\n[RESPONSE_END]\n", result.response.c_str());
  std::fflush(stdout);
  if (print_stats) {
    std::fprintf(stderr, "[STATS] steps=%d duration=%.6f steps_per_s=%.6f\n", result.steps,
                 duration_sec, steps_per_s);
    std::fflush(stderr);
  }
}

int run_single_shot(const char* checkpoint_path, const char* tokenizer_path, const std::string& prompt,
                    int max_steps, float temperature) {
  configure_rope_theta_env(checkpoint_path, tokenizer_path);
  KLLM_MODEL_CLASS model(detect_tokenizer_type(tokenizer_path), tokenizer_path, checkpoint_path,
                         false);
  model.set_sampling_temperature(temperature);
  if (!init_model(model)) {
    return -1;
  }

  const auto start = std::chrono::steady_clock::now();
  GenerationResult result = generate(model, prompt, max_steps);
  const auto end = std::chrono::steady_clock::now();
  const auto duration = std::chrono::duration<double>(end - start).count();
  print_result(result, duration, true);
  return 0;
}

int run_serve(const char* checkpoint_path, const char* tokenizer_path, int max_steps,
              float temperature) {
  configure_rope_theta_env(checkpoint_path, tokenizer_path);
  KLLM_MODEL_CLASS model(detect_tokenizer_type(tokenizer_path), tokenizer_path, checkpoint_path,
                         false);
  model.set_sampling_temperature(temperature);
  if (!init_model(model)) {
    return -1;
  }

  std::cout << "[READY]" << std::endl;
  std::cout.flush();

  ServeState state;
  std::thread stdin_reader([&state]() {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "[EXIT]") {
        state.cancel_requested.store(true);
        state.exit_requested.store(true);
        state.cv.notify_all();
        return;
      }
      if (line == "[CANCEL]") {
        state.cancel_requested.store(true);
        continue;
      }
      if (line != "[PROMPT_START]") {
        continue;
      }

      state.cancel_requested.store(false);
      std::string prompt;
      bool has_end = false;
      while (std::getline(std::cin, line)) {
        if (line == "[PROMPT_END]") {
          has_end = true;
          break;
        }
        if (!prompt.empty()) {
          prompt.push_back('\n');
        }
        prompt += line;
      }

      if (!has_end) {
        state.cancel_requested.store(true);
        state.exit_requested.store(true);
        state.cv.notify_all();
        return;
      }

      {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.prompts.push(prompt);
      }
      state.cv.notify_one();
    }
    state.cancel_requested.store(true);
    state.exit_requested.store(true);
    state.cv.notify_all();
  });

  while (true) {
    std::string prompt;
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      state.cv.wait(lock, [&state]() { return state.exit_requested.load() || !state.prompts.empty(); });
      if (state.prompts.empty()) {
        if (state.exit_requested.load()) {
          break;
        }
        continue;
      }
      prompt = std::move(state.prompts.front());
      state.prompts.pop();
    }

    const auto start = std::chrono::steady_clock::now();
    GenerationResult result = generate(model, prompt, max_steps, &state.cancel_requested);
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();
    if (result.cancelled) {
      std::cout << "[CANCELLED]" << std::endl;
      std::cout.flush();
    }
    print_result(result, duration, false);
  }

  if (stdin_reader.joinable()) {
    stdin_reader.join();
  }
  return 0;
}

void print_usage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program
            << " <checkpoint_path> <tokenizer_path> <prompt> [max_new_tokens] [temperature]\n"
            << "  " << program
            << " --serve <checkpoint_path> <tokenizer_path> [max_new_tokens] [temperature]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = true;
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);

  if (argc >= 2 && std::string(argv[1]) == "--serve") {
    if (argc < 4 || argc > 6) {
      print_usage(argv[0]);
      return -1;
    }
    const char* checkpoint_path = argv[2];
    const char* tokenizer_path = argv[3];
    const int max_steps = argc >= 5 ? parse_max_steps(argv[4]) : 128;
    const float temperature = argc == 6 ? parse_temperature(argv[5]) : 0.0f;
    return run_serve(checkpoint_path, tokenizer_path, max_steps, temperature);
  }

  if (argc < 4 || argc > 6) {
    print_usage(argv[0]);
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];
  const std::string prompt = argv[3];
  const int max_steps = argc >= 5 ? parse_max_steps(argv[4]) : 128;
  const float temperature = argc == 6 ? parse_temperature(argv[5]) : 0.0f;
  return run_single_shot(checkpoint_path, tokenizer_path, prompt, max_steps, temperature);
}
