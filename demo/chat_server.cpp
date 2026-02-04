#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include "base/base.h"

#ifdef QWEN2_SUPPORT
#include "model/qwen2.h"
using ModelType = model::Qwen2Model;
#elif QWEN3_SUPPORT
#include "model/qwen3.h"
using ModelType = model::Qwen3Model;
#else
#include "model/llama3.h"
using ModelType = model::LLama2Model;
#endif

std::unique_ptr<ModelType> g_model;

std::string format_chatml(const std::string& user_input, const std::vector<std::pair<std::string, std::string>>& history) {
  std::string prompt;
  prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  for (const auto& pair : history) {
    prompt += "<|im_start|>user\n" + pair.first + "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n" + pair.second + "<|im_end|>\n";
  }
  prompt += "<|im_start|>user\n" + user_input + "<|im_end|>\n";
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

std::string format_llama(const std::string& user_input, const std::vector<std::pair<std::string, std::string>>& history) {
  std::string prompt;
  for (const auto& pair : history) {
    prompt += "User: " + pair.first + "\n";
    prompt += "Assistant: " + pair.second + "\n";
  }
  prompt += "User: " + user_input + "\n";
  prompt += "Assistant:";
  return prompt;
}

std::string generate_response(const std::string& prompt, int max_length = 512) {
  auto tokens = g_model->encode(prompt);
  int32_t prompt_len = tokens.size();
  if (tokens.empty()) return "[错误] 输入编码失败";

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  const auto& prompt_embedding = g_model->embedding(tokens);
  tensor::Tensor pos_tensor = g_model->get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  words.push_back(next);

  while (pos < max_length) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = g_model->fill_input(pos_tensor, prompt_embedding, is_prompt);
      g_model->predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      std::vector<int32_t> current_tokens = {next};
      const auto& token_embedding = g_model->embedding(current_tokens);
      tensor::Tensor input = g_model->fill_input(pos_tensor, token_embedding, is_prompt);
      g_model->predict(input, pos_tensor, is_prompt, next);
    }
    if (g_model->is_sentence_ending(next)) break;
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
      // 检查是否生成了结束标记
      if (words.size() >= 4) {
        auto decoded = g_model->decode(std::vector<int32_t>(words.end() - 4, words.end()));
        if (decoded.find("<|im_end|>") != std::string::npos ||
            decoded.find("<|endoftext|>") != std::string::npos ||
            decoded.find("</s>") != std::string::npos) {
          break;
        }
      }
    }
    pos += 1;
  }

  // 提取生成的回复，去除 prompt 部分
  std::vector<int32_t> response_tokens(words.begin() + prompt_len, words.end());
  std::string response = g_model->decode(response_tokens);

  // 移除结束标记
  std::vector<std::string> end_markers = {"<|im_end|>", "<|endoftext|>", "</s>", "<|end|>"};
  for (const auto& marker : end_markers) {
    size_t pos = response.find(marker);
    if (pos != std::string::npos) response = response.substr(0, pos);
  }
  size_t start = response.find_first_not_of(" \t\n\r");
  size_t end = response.find_last_not_of(" \t\n\r");
  if (start != std::string::npos && end != std::string::npos) response = response.substr(start, end - start + 1);
  return response;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "用法: " << argv[0] << " <模型路径> <分词器路径>" << std::endl;
    return -1;
  }

  FLAGS_logtostderr = false;
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);

#ifdef QWEN2_SUPPORT
  g_model = std::make_unique<ModelType>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], false);
#elif QWEN3_SUPPORT
  g_model = std::make_unique<ModelType>(base::TokenizerType::kEncodeBpe, argv[2], argv[1], false);
#else
  g_model = std::make_unique<ModelType>(base::TokenizerType::kEncodeSpe, argv[2], argv[1], false);
#endif

  auto init_status = g_model->init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    std::cerr << "[ERROR] 模型初始化失败" << std::endl;
    return -1;
  }

  std::cout << "[READY]" << std::endl;
  std::cout.flush();

  std::string line;
  std::vector<std::pair<std::string, std::string>> history;
  
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (line == "[CLEAR]") {
      history.clear();
      std::cout << "[CLEARED]" << std::endl;
      std::cout.flush();
      continue;
    }
    if (line == "[EXIT]") break;

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    std::string prompt = format_chatml(line, history);
#else
    std::string prompt = format_llama(line, history);
#endif
    
    std::string response = generate_response(prompt, 512);
    history.push_back({line, response});
    if (history.size() > 10) history.erase(history.begin());
    
    std::cout << "[RESPONSE_START]" << std::endl;
    std::cout << response << std::endl;
    std::cout << "[RESPONSE_END]" << std::endl;
    std::cout.flush();
  }
  return 0;
}
