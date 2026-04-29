// 文件说明：RMSNorm 层声明，用于 Transformer block 的归一化计算。

#ifndef KUIPER_INCLUDE_OP_RMSNORM_H_
#define KUIPER_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op {
// RMSNormLayer 支持一维向量和按最后一维归一化的二维/多维视图。
// Qwen3 的 query/key norm 会把 [head_num * head_size] reshape 为 [head_num, head_size]。
class RmsNormLayer : public LayerParam {
 public:
  explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_RMSNORM_H_
