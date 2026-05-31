// 文件说明：SmolLM 模型适配声明，支持轻量模型的权重加载和推理流程。

#ifndef SRC_INCLUDE_MODEL_SMOLLM_H_
#define SRC_INCLUDE_MODEL_SMOLLM_H_

#include "llama3.h"

namespace model {

class SmolLMModel : public LLama2Model {
 public:
  explicit SmolLMModel(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // SRC_INCLUDE_MODEL_SMOLLM_H_
