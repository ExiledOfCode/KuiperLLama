// 文件说明：DeepSeek-Qwen 模型适配声明，复用 Qwen 架构并处理蒸馏模型差异。

#ifndef SRC_INCLUDE_MODEL_DEEPSEEK_QWEN_H_
#define SRC_INCLUDE_MODEL_DEEPSEEK_QWEN_H_

#include "qwen2.h"

namespace model {

class DeepSeekQwenModel : public Qwen2Model {
 public:
  explicit DeepSeekQwenModel(base::TokenizerType tokenizer_type, std::string token_path,
                             std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // SRC_INCLUDE_MODEL_DEEPSEEK_QWEN_H_
