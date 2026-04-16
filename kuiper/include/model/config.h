#ifndef KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#include "base/base.h"
namespace model {
constexpr uint32_t kModelFileMagic = 0x4B4D444C;  // "KMDL"
constexpr uint32_t kModelFileVersion = 1;

struct ModelFileHeader {
  uint32_t magic = kModelFileMagic;
  uint32_t version = kModelFileVersion;
  int32_t weight_type = static_cast<int32_t>(base::WeightType::kWeightTypeFp32);
  int32_t reserved = 0;
};

struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;
#endif
};

struct TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t vocab_size_ = 0;

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
#endif  // KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
