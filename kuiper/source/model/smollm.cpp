#include "model/smollm.h"

namespace model {

SmolLMModel::SmolLMModel(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : LLama2Model(tokenizer_type, token_path, model_path, is_quant_model) {}

}  // namespace model
