// 文件说明：RoPE 层声明，为 Q/K 向量注入旋转位置编码。

#ifndef SRC_INCLUDE_OP_ROPE_H_
#define SRC_INCLUDE_OP_ROPE_H_
#include "layer.h"
namespace op {
// RoPELayer 对 query 和 key 原地应用旋转位置编码。
// input_pos 固定放在 CPU 上，因为它只是一个标量位置，CUDA kernel 会读取该位置值。
class RoPELayer : public Layer {
 public:
  explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;        // query 总维度。
  int32_t kv_dim_ = 0;     // key 总维度。
  int32_t head_size_ = 0;  // 每个 head 中参与旋转的维度。
};
}  // namespace op
#endif  // SRC_INCLUDE_OP_ROPE_H_
