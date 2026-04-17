#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include "model/qwen2.h"

namespace {

using json = nlohmann::json;
constexpr size_t kTraceTokenPreviewLimit = 256;
constexpr size_t kTraceTextPreviewLimit = 4096;
constexpr int kTraceSamplingPreviewLimit = 128;

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

double elapsed_ms(const std::chrono::steady_clock::time_point& begin,
                  const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool is_continuation_byte(unsigned char value) { return (value & 0xC0U) == 0x80U; }

std::string sanitize_utf8(const std::string& text) {
  std::string output;
  output.reserve(text.size());

  const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
  const size_t size = text.size();
  size_t index = 0;
  while (index < size) {
    const unsigned char ch = bytes[index];
    if (ch <= 0x7FU) {
      output.push_back(static_cast<char>(ch));
      ++index;
      continue;
    }

    size_t width = 0;
    if ((ch & 0xE0U) == 0xC0U) {
      if (ch >= 0xC2U) {
        width = 2;
      }
    } else if ((ch & 0xF0U) == 0xE0U) {
      width = 3;
    } else if ((ch & 0xF8U) == 0xF0U) {
      if (ch <= 0xF4U) {
        width = 4;
      }
    }

    if (width == 0 || index + width > size) {
      output.push_back('?');
      ++index;
      continue;
    }

    bool valid = true;
    for (size_t offset = 1; offset < width; ++offset) {
      if (!is_continuation_byte(bytes[index + offset])) {
        valid = false;
        break;
      }
    }
    if (valid && width == 3) {
      const unsigned char c1 = bytes[index + 1];
      if ((ch == 0xE0U && c1 < 0xA0U) || (ch == 0xEDU && c1 >= 0xA0U)) {
        valid = false;
      }
    }
    if (valid && width == 4) {
      const unsigned char c1 = bytes[index + 1];
      if ((ch == 0xF0U && c1 < 0x90U) || (ch == 0xF4U && c1 >= 0x90U)) {
        valid = false;
      }
    }

    if (!valid) {
      output.push_back('?');
      ++index;
      continue;
    }

    output.append(text, index, width);
    index += width;
  }

  return output;
}

bool trace_enabled_from_env() {
  const char* raw = std::getenv("KLLM_TRACE_ENABLED");
  if (raw == nullptr) {
    return true;
  }
  std::string text = raw;
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return !(text == "0" || text == "false" || text == "off" || text == "no");
}

std::string trim(const std::string& text) {
  const auto start = text.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(start, end - start + 1);
}

std::string truncate_text(const std::string& text, size_t limit) {
  const std::string safe_text = sanitize_utf8(text);
  if (safe_text.size() <= limit) {
    return safe_text;
  }
  if (limit <= 3) {
    return sanitize_utf8(safe_text.substr(0, limit));
  }
  return sanitize_utf8(safe_text.substr(0, limit - 3)) + "...";
}

std::string escape_inline_text(const std::string& text, size_t limit = 48) {
  const std::string safe_text = sanitize_utf8(text);
  std::string output;
  output.reserve(std::min(safe_text.size(), limit) + 8);
  for (char ch : safe_text) {
    if (ch == '\n') {
      output += "\\n";
    } else if (ch == '\r') {
      continue;
    } else if (ch == '\t') {
      output += "\\t";
    } else {
      output.push_back(ch);
    }
    if (output.size() >= limit) {
      break;
    }
  }
  return trim(sanitize_utf8(output));
}

void emit_trace_json(const json& payload, bool enabled) {
  if (!enabled) {
    return;
  }
  std::cout << "[TRACE]" << payload.dump() << std::endl;
  std::cout.flush();
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

std::vector<std::string> build_token_pieces_preview(const model::Qwen2Model& model,
                                                    const std::vector<int32_t>& token_ids,
                                                    size_t limit) {
  const size_t count = std::min(limit, token_ids.size());
  std::vector<std::string> pieces;
  pieces.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const std::string piece = escape_inline_text(model.decode(std::vector<int32_t>{token_ids[i]}), 64);
    pieces.push_back(piece.empty() ? ("<id:" + std::to_string(token_ids[i]) + ">") : piece);
  }
  return pieces;
}

std::string remove_end_markers(const std::string& text) {
  std::string output = text;
  const std::vector<std::string> end_markers = {
      "<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"};

  for (const auto& marker : end_markers) {
    const auto marker_pos = output.find(marker);
    if (marker_pos != std::string::npos) {
      output = output.substr(0, marker_pos);
    }
  }
  return trim(output);
}

std::string collapse_repeated_lines(const std::string& text) {
  std::istringstream input(text);
  std::string line;
  std::vector<std::string> kept;

  auto is_structured_tag_line = [](const std::string& value) {
    return !value.empty() && value.front() == '<' && value.back() == '>' && value.find(' ') == std::string::npos;
  };

  std::string prev_norm;
  int same_run = 0;
  while (std::getline(input, line)) {
    const std::string norm = trim(line);
    if (!norm.empty() && norm == prev_norm && norm.size() <= 48) {
      same_run += 1;
      if (is_structured_tag_line(norm)) {
        continue;
      }
      // 短重复行（如 "orean"）直接截断尾部，并移除第一条重复短行
      if (same_run >= 2 && norm.size() <= 32) {
        if (!kept.empty() && trim(kept.back()) == norm) {
          kept.pop_back();
        }
        break;
      }
      if (same_run >= 3) {
        continue;
      }
    } else {
      prev_norm = norm;
      same_run = 1;
    }
    kept.push_back(line);
  }

  std::string output;
  for (size_t i = 0; i < kept.size(); ++i) {
    output += kept[i];
    if (i + 1 < kept.size()) {
      output.push_back('\n');
    }
  }
  return trim(output);
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

void reset_generation_state(const model::Qwen2Model& model) {
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
    clear_tensor_buffer(model.get_buffer(buffer_type));
  }
}

GenerationResult generate(const model::Qwen2Model& model, const std::string& sentence,
                          int max_new_tokens, bool emit_trace,
                          const std::atomic<bool>* cancel_requested = nullptr) {
  reset_generation_state(model);
  model.set_profile_enabled(emit_trace);
  const auto step1_begin = std::chrono::steady_clock::now();
  auto tokens = model.encode(sentence);
  const auto step1_end = std::chrono::steady_clock::now();
  const double tokenization_ms = elapsed_ms(step1_begin, step1_end);
  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  if (tokens.empty()) {
    return {"", 0};
  }
  const int32_t max_seq_len = model.max_seq_len();
  const int32_t requested_max_new_tokens = std::max(1, max_new_tokens);
  if (max_seq_len > 0 && prompt_len >= max_seq_len) {
    std::ostringstream error;
    error << "当前请求超过模型上下文长度：prompt_tokens=" << prompt_len
          << ", max_token=" << requested_max_new_tokens << ", seq_len=" << max_seq_len
          << "。请减少历史轮数或降低 max_token。";
    return {error.str(), 0};
  }
  if (max_seq_len > 0 && prompt_len + requested_max_new_tokens > max_seq_len) {
    std::ostringstream error;
    error << "当前请求超过模型上下文长度：prompt_tokens=" << prompt_len
          << ", max_token=" << requested_max_new_tokens << ", seq_len=" << max_seq_len
          << "。请减少历史轮数或降低 max_token。";
    return {error.str(), 0};
  }

  const auto step2_begin = std::chrono::steady_clock::now();
  const auto token_pieces_preview =
      build_token_pieces_preview(model, tokens, kTraceTokenPreviewLimit);
  std::vector<int32_t> token_ids_preview;
  token_ids_preview.reserve(std::min(tokens.size(), kTraceTokenPreviewLimit));
  for (size_t i = 0; i < std::min(tokens.size(), kTraceTokenPreviewLimit); ++i) {
    token_ids_preview.push_back(tokens[i]);
  }
  const auto step2_end = std::chrono::steady_clock::now();
  const double encoding_ms = elapsed_ms(step2_begin, step2_end);
  emit_trace_json(
      json{{"event", "step"},
           {"step", "tokenization"},
           {"title", "Step1: Tokenization"},
           {"duration_ms", tokenization_ms},
           {"input_text", truncate_text(sentence, kTraceTextPreviewLimit)},
           {"token_count", prompt_len},
           {"tokens_preview", token_pieces_preview},
           {"truncated", tokens.size() > token_pieces_preview.size()}},
      emit_trace);
  emit_trace_json(
      json{{"event", "step"},
           {"step", "encoding"},
           {"title", "Step2: Encoding"},
           {"duration_ms", encoding_ms},
           {"token_count", prompt_len},
           {"token_ids_preview", token_ids_preview},
           {"truncated", tokens.size() > token_ids_preview.size()}},
      emit_trace);
  emit_trace_json(
      json{{"event", "step"},
           {"step", "transformer"},
           {"title", "Step3: Transformer Inference"},
           {"operations", {"attention", "hidden_states", "logits"}},
           {"status", "running"}},
      emit_trace);

  if (emit_trace) {
    model.reset_profile_stats();
  }
  const int32_t total_steps = prompt_len + requested_max_new_tokens;
  const auto step3_begin = std::chrono::steady_clock::now();

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  int32_t last_generated = -1;
  int32_t same_token_run = 0;
  int32_t generated_token_count = 0;
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
    // 仅在生成阶段检查结束标记，避免在 prompt 阶段被提前截断
    if (!is_prompt && model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      generated_token_count += 1;
      words.push_back(next);
      if (generated_token_count <= kTraceSamplingPreviewLimit) {
        const std::string selected_token =
            escape_inline_text(model.decode(std::vector<int32_t>{next}), 64);
        emit_trace_json(
            json{{"event", "sampling"},
                 {"step", "sampling"},
                 {"title", "Step4: Sampling"},
                 {"sampler", "argmax"},
                 {"sample_index", generated_token_count},
                 {"selected_token_id", next},
                 {"selected_token", selected_token.empty() ? "<blank>" : selected_token}},
            emit_trace);
      }

      // 1) 连续同 token 重复截断，避免出现 "orean orea..." 这类刷屏
      if (next == last_generated) {
        same_token_run += 1;
      } else {
        last_generated = next;
        same_token_run = 1;
      }
      if (same_token_run >= 24) {
        break;
      }

      // 2) 检查最近 token 是否出现明确结束标记
      if (words.size() > static_cast<size_t>(prompt_len + 6)) {
        std::vector<int32_t> tail_tokens(words.end() - 6, words.end());
        const std::string tail_text = model.decode(tail_tokens);
        if (tail_text.find("<|im_end|>") != std::string::npos ||
            tail_text.find("<|endoftext|>") != std::string::npos ||
            tail_text.find("</s>") != std::string::npos ||
            tail_text.find("<|end|>") != std::string::npos) {
          break;
        }
      }

      // 3) 最近窗口完全重复（例如 12 token 周期重复）时截断
      constexpr int kRepeatWindow = 12;
      if (words.size() >= static_cast<size_t>(prompt_len + kRepeatWindow * 2)) {
        bool repeated = true;
        auto end_it = words.end();
        for (int i = 0; i < kRepeatWindow; ++i) {
          if (*(end_it - 1 - i) != *(end_it - 1 - kRepeatWindow - i)) {
            repeated = false;
            break;
          }
        }
        if (repeated) {
          break;
        }
      }
    }
    pos += 1;
  }
  const auto step3_end = std::chrono::steady_clock::now();
  const double transformer_wall_ms = elapsed_ms(step3_begin, step3_end);
  const std::vector<model::OpProfileStat> op_profile_stats =
      emit_trace ? model.get_profile_stats() : std::vector<model::OpProfileStat>{};

  std::string response;
  std::vector<int32_t> response_tokens;
  const auto step5_begin = std::chrono::steady_clock::now();
  if (words.size() > static_cast<size_t>(prompt_len)) {
    response_tokens = std::vector<int32_t>(words.begin() + prompt_len, words.end());
    response = collapse_repeated_lines(remove_end_markers(sanitize_utf8(model.decode(response_tokens))));
  }
  const auto step5_end = std::chrono::steady_clock::now();
  const double decode_ms = elapsed_ms(step5_begin, step5_end);

  double total_op_ms = 0.0;
  double sampling_ms = 0.0;
  json operator_profile = json::array();
  for (const auto& stat : op_profile_stats) {
    total_op_ms += stat.total_ms;
    if (stat.op_name == "sampler.argmax") {
      sampling_ms += stat.total_ms;
    }
    operator_profile.push_back(json{{"name", stat.op_name},
                                    {"total_ms", stat.total_ms},
                                    {"calls", stat.calls},
                                    {"avg_ms", stat.avg_ms}});
  }
  const double transformer_ms = std::max(0.0, total_op_ms - sampling_ms);
  std::vector<int32_t> generated_preview_ids;
  generated_preview_ids.reserve(std::min(response_tokens.size(), kTraceTokenPreviewLimit));
  for (size_t i = 0; i < std::min(response_tokens.size(), kTraceTokenPreviewLimit); ++i) {
    generated_preview_ids.push_back(response_tokens[i]);
  }
  emit_trace_json(
      json{{"event", "step"},
           {"step", "transformer"},
           {"title", "Step3: Transformer Inference"},
           {"status", cancelled ? "cancelled" : "done"},
           {"duration_ms", transformer_ms > 0.0 ? transformer_ms : transformer_wall_ms},
           {"operator_count", static_cast<int32_t>(op_profile_stats.size())},
           {"operator_profile", operator_profile}},
      emit_trace);
  emit_trace_json(
      json{{"event", "step"},
           {"step", "sampling"},
           {"title", "Step4: Sampling"},
           {"sampler", "argmax"},
           {"duration_ms", sampling_ms},
           {"generated_token_count", generated_token_count},
           {"preview_token_ids", generated_preview_ids},
           {"truncated", response_tokens.size() > generated_preview_ids.size()}},
      emit_trace);
  emit_trace_json(
      json{{"event", "step"},
           {"step", "decode"},
           {"title", "Step5: Decode"},
           {"duration_ms", decode_ms},
           {"generated_text_preview", truncate_text(response, kTraceTextPreviewLimit)},
           {"generated_char_count", static_cast<int32_t>(response.size())}},
      emit_trace);
  return {response, std::min(pos, total_steps), cancelled};
}

bool init_model(model::Qwen2Model& model) {
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
  const double steps_per_s = duration_sec > 0 ? static_cast<double>(result.steps) / duration_sec : 0.0;
  std::printf("[RESPONSE_START]\n%s\n[RESPONSE_END]\n", result.response.c_str());
  std::fflush(stdout);
  if (print_stats) {
    std::fprintf(stderr, "[STATS] steps=%d duration=%.6f steps_per_s=%.6f\n", result.steps, duration_sec,
                 steps_per_s);
    std::fflush(stderr);
  }
}

int run_single_shot(const char* checkpoint_path, const char* tokenizer_path, const std::string& prompt,
                    int max_steps, float temperature) {
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  model.set_sampling_temperature(temperature);
  if (!init_model(model)) {
    return -1;
  }

  const bool emit_trace = trace_enabled_from_env();
  const auto start = std::chrono::steady_clock::now();
  GenerationResult result = generate(model, prompt, max_steps, emit_trace);
  const auto end = std::chrono::steady_clock::now();
  const auto duration = std::chrono::duration<double>(end - start).count();
  print_result(result, duration, true);
  return 0;
}

int run_serve(const char* checkpoint_path, const char* tokenizer_path, int max_steps,
              float temperature) {
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  model.set_sampling_temperature(temperature);
  if (!init_model(model)) {
    return -1;
  }

  const bool emit_trace = trace_enabled_from_env();
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
    GenerationResult result = generate(model, prompt, max_steps, emit_trace, &state.cancel_requested);
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();
    emit_trace_json(
        json{{"event", "done"},
             {"step", "done"},
             {"title", "Inference Done"},
             {"duration_seconds", duration},
             {"generated_steps", result.steps},
             {"state", result.cancelled ? "cancelled" : "completed"}},
        emit_trace);
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
