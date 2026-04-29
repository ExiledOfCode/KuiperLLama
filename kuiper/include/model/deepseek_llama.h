// 文件说明：DeepSeek-Llama 模型适配声明，复用 Llama 架构并处理蒸馏模型差异。

#ifndef KUIPER_INCLUDE_MODEL_DEEPSEEK_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_DEEPSEEK_LLAMA_H_

#include "llama3.h"

namespace model {

class DeepSeekLlamaModel : public LLama2Model {
 public:
  explicit DeepSeekLlamaModel(base::TokenizerType tokenizer_type, std::string token_path,
                              std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_DEEPSEEK_LLAMA_H_
