// 文件说明：Qwen3 demo 入口，支持推理跟踪、采样参数和服务端流式输出标记。

#include <atomic>
#include <algorithm>
#include <cctype>
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
#include <base/tick.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include "model/qwen3.h"

namespace {

using json = nlohmann::json;
constexpr size_t kTraceTokenPreviewLimit = 256;
constexpr size_t kTraceTextPreviewLimit = 4096;
constexpr int kTraceSamplingPreviewLimit = 128;

struct GenerationResult {
  std::string response;  // 清理 stop marker 和重复行后的最终文本。
  int32_t steps = 0;     // 实际执行的 token step 数，用于吞吐统计。
  bool cancelled = false;
  std::string finish_reason = "unknown";  // length/eos/cancelled/repeat 等结束原因。
};

struct ServeState {
  // stdin_reader 线程负责解析控制帧，主线程负责串行推理；两者通过队列和条件变量交接 prompt。
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
  // 模型可能吐出半个 UTF-8 字符或非法 byte，这里替换为 '?'，避免 JSON/终端输出被破坏。
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
  // 默认打开 trace，前端可以通过 KLLM_TRACE_ENABLED=0 降低 stdout 事件量。
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
  // trace 中的 token preview 需要单行展示，因此折叠换行和制表符。
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
  // 所有机器可解析事件都带固定前缀，服务端/前端按行读取时无需猜测普通文本边界。
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

void emit_response_chunk_json(const std::string& text) {
  std::cout << "[RESPONSE_CHUNK]" << json{{"text", sanitize_utf8(text)}}.dump() << std::endl;
  std::cout.flush();
}

std::vector<std::string> build_token_pieces_preview(const model::Qwen3Model& model,
                                                    const std::vector<int32_t>& token_ids,
                                                    size_t limit) {
  // 只 decode 前 limit 个 token，避免长上下文 prompt 产生过大的 trace payload。
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
  // 某些 tokenizer 会把特殊结束符 decode 成可见文本，最终响应中需要裁掉。
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
  // 对 demo 输出做轻量后处理：短行重复太多通常是退化生成，保留有限次数即可。
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
  // 服务模式下同一个模型处理多轮请求；清理运行时 buffer 可避免上轮残留影响调试和 trace。
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

void configure_rope_theta_env(const char* checkpoint_path, const char* tokenizer_path) {
  // Qwen 系列 rope_theta 可能在 config.json 中；通过环境变量传给 RoPE kernel，保持 C++ 接口不变。
  namespace fs = std::filesystem;
  std::vector<fs::path> candidates;
  if (checkpoint_path != nullptr && *checkpoint_path != '\0') {
    candidates.emplace_back(fs::path(checkpoint_path).parent_path() / "config.json");
  }
  if (tokenizer_path != nullptr && *tokenizer_path != '\0') {
    const fs::path tokenizer_dir = fs::path(tokenizer_path).parent_path() / "config.json";
    if (std::find(candidates.begin(), candidates.end(), tokenizer_dir) == candidates.end()) {
      candidates.push_back(tokenizer_dir);
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

void reset_generation_state(const model::Qwen3Model& model) {
  using model::ModelBufferType;
  // 只重置运行时中间结果，不清理权重、RoPE cache 或 tokenizer。
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

GenerationResult generate(const model::Qwen3Model& model, const std::string& sentence,
                          int max_new_tokens, bool emit_trace,
                          const std::atomic<bool>* cancel_requested = nullptr) {
  // 生成流程分为五个 trace step：
  // 1 tokenization，2 token id preview，3 transformer，4 sampling，5 decode。
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
    // prompt 已经占满上下文时直接返回中文错误，避免后续 KV cache 越界。
    std::ostringstream error;
    error << "当前请求超过模型上下文长度：prompt_tokens=" << prompt_len
          << ", max_token=" << requested_max_new_tokens << ", seq_len=" << max_seq_len
          << "。请减少历史轮数或降低 max_token。";
    return {error.str(), 0};
  }
  int32_t effective_max_new_tokens = requested_max_new_tokens;
  if (max_seq_len > 0 && prompt_len + requested_max_new_tokens > max_seq_len) {
    // 保守截断生成长度，使 prompt + decode token 始终落在模型 seq_len 内。
    effective_max_new_tokens = std::max(1, max_seq_len - prompt_len);
    std::fprintf(stderr,
                 "[WARN] max_new_tokens clamped from %d to %d because prompt_tokens=%d and "
                 "seq_len=%d\n",
                 requested_max_new_tokens, effective_max_new_tokens, prompt_len, max_seq_len);
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
  const int32_t total_steps = prompt_len + effective_max_new_tokens - 1;
  const auto step3_begin = std::chrono::steady_clock::now();

  int32_t pos = 0;
  int32_t next = tokens.at(pos);
  bool is_prompt = true;
  int32_t last_generated = -1;
  int32_t same_token_run = 0;
  int32_t generated_token_count = 0;
  bool cancelled = false;
  std::string finish_reason = "length";
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  words.push_back(next);
  while (pos < total_steps) {
    if (cancel_requested != nullptr && cancel_requested->load()) {
      cancelled = true;
      finish_reason = "cancelled";
      break;
    }
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      // prefill：喂入 prompt 中当前位置的 embedding，只更新 KV cache，不采样。
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      // decode：把上一轮采样得到的 token 重新 embedding，再预测下一个 token。
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (cancel_requested != nullptr && cancel_requested->load()) {
      cancelled = true;
      finish_reason = "cancelled";
      break;
    }

    if (!is_prompt && model.is_sentence_ending(next)) {
      finish_reason = "eos";
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

      if (next == last_generated) {
        same_token_run += 1;
      } else {
        last_generated = next;
        same_token_run = 1;
      }
      if (same_token_run >= 24) {
        // 防止 argmax 或坏权重导致同一个 token 无限重复。
        finish_reason = "same_token_repeat";
        break;
      }

      if (words.size() > static_cast<size_t>(prompt_len + 6)) {
        std::vector<int32_t> tail_tokens(words.end() - 6, words.end());
        const std::string tail_text = model.decode(tail_tokens);
        if (tail_text.find("<|im_end|>") != std::string::npos ||
            tail_text.find("<|endoftext|>") != std::string::npos ||
            tail_text.find("</s>") != std::string::npos ||
            tail_text.find("<|end|>") != std::string::npos) {
          finish_reason = "stop_marker";
          break;
        }
      }

      constexpr int kRepeatWindow = 12;
      if (words.size() >= static_cast<size_t>(prompt_len + kRepeatWindow * 2)) {
        // 检测最近两个固定窗口是否完全相同，拦截循环句段。
        bool repeated = true;
        auto end_it = words.end();
        for (int i = 0; i < kRepeatWindow; ++i) {
          if (*(end_it - 1 - i) != *(end_it - 1 - kRepeatWindow - i)) {
            repeated = false;
            break;
          }
        }
        if (repeated) {
          finish_reason = "window_repeat";
          break;
        }
      }

      const std::string token_piece = sanitize_utf8(model.decode(std::vector<int32_t>{next}));
      if (!token_piece.empty()) {
        // 服务端可边生成边消费 chunk，最终完整响应仍由 print_result 输出。
        emit_response_chunk_json(token_piece);
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
    // words 同时包含 prompt 和生成 token，最终响应只 decode prompt 之后的部分。
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
           {"requested_max_new_tokens", requested_max_new_tokens},
           {"effective_max_new_tokens", effective_max_new_tokens},
           {"finish_reason", finish_reason},
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
  return {response, std::min(pos, total_steps), cancelled, finish_reason};
}

bool init_model(model::Qwen3Model& model) {
  // 初始化阶段把加载进度同步输出给调用方，便于前端显示权重/缓存准备进度。
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
    // demo 允许 CUDA 初始化失败后回退 CPU，便于没有 GPU 的开发环境继续验证流程。
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
    std::fprintf(stderr, "[STATS] steps=%d duration=%.6f steps_per_s=%.6f finish_reason=%s\n",
                 result.steps, duration_sec, steps_per_s, result.finish_reason.c_str());
    std::fflush(stderr);
  }
}

int run_single_shot(const char* checkpoint_path, const char* tokenizer_path, const std::string& prompt,
                    int max_steps, float temperature) {
  // 单次模式：启动、加载、生成、打印统计后退出。
  configure_rope_theta_env(checkpoint_path, tokenizer_path);
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
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
  // 服务模式：模型常驻内存，stdin/stdout 使用简单帧协议和上层服务通信。
  configure_rope_theta_env(checkpoint_path, tokenizer_path);
  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  model.set_sampling_temperature(temperature);
  if (!init_model(model)) {
    return -1;
  }

  const bool emit_trace = trace_enabled_from_env();
  std::cout << "[READY]" << std::endl;
  std::cout.flush();

  ServeState state;
  std::thread stdin_reader([&state]() {
    // 单独线程阻塞读取 stdin，主线程即使在推理中也能收到 CANCEL/EXIT 信号。
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

      // prompt 允许多行内容，直到 PROMPT_END 结束。
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
      // 主线程串行消费 prompt，避免一个 Qwen3Model 实例被并发调用。
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
             {"finish_reason", result.finish_reason},
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
