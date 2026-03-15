#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>

#include "model/qwen2.h"

namespace {

struct GenerationResult {
  std::string response;
  int32_t steps = 0;
};

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

  std::string prev_norm;
  int same_run = 0;
  while (std::getline(input, line)) {
    const std::string norm = trim(line);
    if (!norm.empty() && norm == prev_norm && norm.size() <= 48) {
      same_run += 1;
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

GenerationResult generate(const model::Qwen2Model& model, const std::string& sentence,
                          int max_new_tokens) {
  auto tokens = model.encode(sentence);
  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  if (tokens.empty()) {
    return {"", 0};
  }
  const int32_t total_steps = prompt_len + std::max(1, max_new_tokens);

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  int32_t last_generated = -1;
  int32_t same_token_run = 0;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  words.push_back(next);
  while (pos < total_steps) {
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
    // 仅在生成阶段检查结束标记，避免在 prompt 阶段被提前截断
    if (!is_prompt && model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);

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

  std::string response;
  if (words.size() > static_cast<size_t>(prompt_len)) {
    std::vector<int32_t> response_tokens(words.begin() + prompt_len, words.end());
    response = collapse_repeated_lines(remove_end_markers(model.decode(response_tokens)));
  }
  return {response, std::min(pos, total_steps)};
}

bool init_model(model::Qwen2Model& model) {
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
                    int max_steps) {
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
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

int run_serve(const char* checkpoint_path, const char* tokenizer_path, int max_steps) {
  model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  if (!init_model(model)) {
    return -1;
  }

  std::cout << "[READY]" << std::endl;
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "[EXIT]") {
      break;
    }
    if (line != "[PROMPT_START]") {
      continue;
    }

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
      break;
    }

    const auto start = std::chrono::steady_clock::now();
    GenerationResult result = generate(model, prompt, max_steps);
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();
    print_result(result, duration, false);
  }
  return 0;
}

void print_usage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program
            << " <checkpoint_path> <tokenizer_path> <prompt> [max_new_tokens]\n"
            << "  " << program << " --serve <checkpoint_path> <tokenizer_path> [max_new_tokens]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = true;
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);

  if (argc >= 2 && std::string(argv[1]) == "--serve") {
    if (argc < 4 || argc > 5) {
      print_usage(argv[0]);
      return -1;
    }
    const char* checkpoint_path = argv[2];
    const char* tokenizer_path = argv[3];
    const int max_steps = argc == 5 ? parse_max_steps(argv[4]) : 128;
    return run_serve(checkpoint_path, tokenizer_path, max_steps);
  }

  if (argc < 4 || argc > 5) {
    print_usage(argv[0]);
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];
  const std::string prompt = argv[3];
  const int max_steps = argc == 5 ? parse_max_steps(argv[4]) : 128;
  return run_single_shot(checkpoint_path, tokenizer_path, prompt, max_steps);
}
