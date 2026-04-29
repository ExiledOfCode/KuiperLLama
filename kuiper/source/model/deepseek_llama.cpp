// 文件说明：DeepSeek-Llama 模型实现，组织层创建、权重绑定和 token 生成流程。

#include "model/deepseek_llama.h"

namespace model {

DeepSeekLlamaModel::DeepSeekLlamaModel(base::TokenizerType tokenizer_type, std::string token_path,
                                       std::string model_path, bool is_quant_model)
    : LLama2Model(tokenizer_type, token_path, model_path, is_quant_model) {}

}  // namespace model
