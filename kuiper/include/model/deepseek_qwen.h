#ifndef KUIPER_INCLUDE_MODEL_DEEPSEEK_QWEN_H_
#define KUIPER_INCLUDE_MODEL_DEEPSEEK_QWEN_H_

#include "qwen2.h"

namespace model {

class DeepSeekQwenModel : public Qwen2Model {
 public:
  explicit DeepSeekQwenModel(base::TokenizerType tokenizer_type, std::string token_path,
                             std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_DEEPSEEK_QWEN_H_
