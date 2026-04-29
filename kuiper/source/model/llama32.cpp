// 文件说明：Llama3.2 模型实现，适配 3.2 系列参数并执行端到端推理。

#include "model/llama32.h"

namespace model {

Llama32Model::Llama32Model(base::TokenizerType tokenizer_type, std::string token_path,
                           std::string model_path, bool is_quant_model)
    : LLama2Model(tokenizer_type, token_path, model_path, is_quant_model) {}

}  // namespace model
