// 文件说明：Llama3.2 模型适配声明，覆盖 3.2 系列配置和生成流程。

#ifndef SRC_INCLUDE_MODEL_LLAMA32_H_
#define SRC_INCLUDE_MODEL_LLAMA32_H_

#include "llama3.h"

namespace model {

class Llama32Model : public LLama2Model {
 public:
  explicit Llama32Model(base::TokenizerType tokenizer_type, std::string token_path,
                        std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // SRC_INCLUDE_MODEL_LLAMA32_H_
