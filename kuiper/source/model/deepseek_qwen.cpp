#include "model/deepseek_qwen.h"

namespace model {

DeepSeekQwenModel::DeepSeekQwenModel(base::TokenizerType tokenizer_type, std::string token_path,
                                     std::string model_path, bool is_quant_model)
    : Qwen2Model(tokenizer_type, token_path, model_path, is_quant_model) {}

}  // namespace model
