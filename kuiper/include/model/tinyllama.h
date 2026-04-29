// 文件说明：TinyLlama 模型适配声明，支持 TinyLlama 的模型结构和生成流程。

#ifndef KUIPER_INCLUDE_MODEL_TINYLLAMA_H_
#define KUIPER_INCLUDE_MODEL_TINYLLAMA_H_

#include "llama3.h"

namespace model {

class TinyLlamaModel : public LLama2Model {
 public:
  explicit TinyLlamaModel(base::TokenizerType tokenizer_type, std::string token_path,
                          std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_TINYLLAMA_H_
