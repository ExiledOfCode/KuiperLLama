#ifndef KUIPER_INCLUDE_MODEL_LLAMA32_H_
#define KUIPER_INCLUDE_MODEL_LLAMA32_H_

#include "llama3.h"

namespace model {

class Llama32Model : public LLama2Model {
 public:
  explicit Llama32Model(base::TokenizerType tokenizer_type, std::string token_path,
                        std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_LLAMA32_H_
