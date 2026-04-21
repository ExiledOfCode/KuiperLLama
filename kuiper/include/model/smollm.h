#ifndef KUIPER_INCLUDE_MODEL_SMOLLM_H_
#define KUIPER_INCLUDE_MODEL_SMOLLM_H_

#include "llama3.h"

namespace model {

class SmolLMModel : public LLama2Model {
 public:
  explicit SmolLMModel(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_SMOLLM_H_
