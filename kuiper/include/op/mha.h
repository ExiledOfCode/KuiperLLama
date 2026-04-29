// 文件说明：多头注意力层声明，封装 QK 打分、KV cache 读取和注意力输出。

#ifndef KUIPER_INLCUDE_MHA_H
#define KUIPER_INLCUDE_MHA_H
#include <base/cuda_config.h>
#include "layer.h"
namespace op {
// MultiHeadAttention 负责单 token decode 的注意力计算。
// 输入约定：
// 0 query，1 score scratch，2 key cache，3 value cache，4 预留；
// 分页 KV cache 开启时，2/3 为空 Tensor，kernel 改用 page table 读取历史 K/V。
class MultiHeadAttention : public op::Layer {
 public:
  explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                              int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                              int32_t head_num, int32_t head_size);

  base::Status check() const override;

  void set_pos(int32_t pos);
  void set_layer_idx(int32_t layer_idx);
  void set_paged_kv_cache(const void* key_page_table, const void* value_page_table,
                          int32_t page_size, bool enabled);

  base::Status forward() override;

 private:
  int32_t layer_index_ = 0;
  int32_t pos_ = 0;
  int32_t kv_mul_ = 0;     // GQA 中一个 kv head 对应多少个 query head。
  int32_t kv_dim_ = 0;     // 单 token K/V 向量总维度。
  int32_t seq_len_ = 0;    // 连续 cache 的最大长度，也是 score scratch 的长度。
  int32_t head_num_ = 0;   // query head 数量。
  int32_t head_size_ = 0;  // 单 head 维度。
  const void* key_page_table_ = nullptr;
  const void* value_page_table_ = nullptr;
  int32_t page_size_ = 0;
  bool paged_kv_cache_enabled_ = false;
};
}  // namespace op
#endif  // KUIPER_INLCUDE_MHA_H
