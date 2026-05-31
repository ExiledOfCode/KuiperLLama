// 文件说明：模型配置结构定义，描述 Transformer 层数、维度、量化和词表信息。

#ifndef SRC_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define SRC_INCLUDE_MODEL_LLAMA_CONFIG_H_
#include "base/base.h"
namespace model {
constexpr uint32_t kModelFileMagic = 0x4B4D444C;  // "KMDL"
constexpr uint32_t kModelFileVersion = 1;

// 导出权重文件的显式文件头。
// 旧版文件没有这个 header，Model::read_model_file 会回退为“文件起始就是 ModelConfig”。
// reserved 当前复用为量化模型中非参数张量的 data type，用于区分 FP32/BF16 归一化权重等。
struct ModelFileHeader {
  uint32_t magic = kModelFileMagic;
  uint32_t version = kModelFileVersion;
  int32_t weight_type = static_cast<int32_t>(base::WeightType::kWeightTypeFp32);
  int32_t reserved = 0;
};

// 二进制模型文件内直接保存的基础配置。
// 这里保持 POD 布局，便于从 mmap/fread 直接读取；派生字段在 TransformerConfig 中补齐。
struct ModelConfig {
  int32_t dim = 0;          // attention 投影维度，通常等于 head_num * head_size。
  int32_t hidden_dim = 0;   // token embedding/残差流维度。
  int32_t layer_num = 0;    // Transformer block 数量。
  int32_t head_num = 0;     // query head 数量。
  int32_t kv_head_num = 0;  // key/value head 数量，GQA/MQA 时小于 head_num。
  int32_t vocab_size = 0;   // 正数表示 lm_head 与 embedding 共享，负数表示独立权重。
  int32_t seq_len = 0;      // 模型支持的最大上下文长度。
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;  // Qwen3 FFN 中间层维度。
#endif
};

// 运行期配置，会在读取 ModelConfig 后计算出 kernel 需要的派生参数。
struct TransformerConfig {
  int32_t kv_dim_ = 0;       // 每个 token 的 key/value 总维度：dim * kv_head_num / head_num。
  int32_t kv_mul_ = 0;       // 每个 kv head 需要服务多少个 query head。
  int32_t head_size_ = 0;    // 单个 attention head 的维度。
  int32_t vocab_size_ = 0;   // 取绝对值后的词表大小，用于 logits 输出。

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;
#endif
};
}  // namespace model
#endif  // SRC_INCLUDE_MODEL_LLAMA_CONFIG_H_
