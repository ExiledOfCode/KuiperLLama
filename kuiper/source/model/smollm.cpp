// 文件说明：SmolLM 模型实现，处理轻量模型的层创建、前向计算和输出解码。

#include "model/smollm.h"

namespace model {

SmolLMModel::SmolLMModel(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : LLama2Model(tokenizer_type, token_path, model_path, is_quant_model) {}

}  // namespace model
