// 文件说明：TinyLlama 模型实现，覆盖 TinyLlama 权重布局和逐 token 推理。

#include "model/tinyllama.h"

namespace model {

TinyLlamaModel::TinyLlamaModel(base::TokenizerType tokenizer_type, std::string token_path,
                               std::string model_path, bool is_quant_model)
    : LLama2Model(tokenizer_type, token_path, model_path, is_quant_model) {}

}  // namespace model
