// 文件说明：残差加法层声明，负责 Transformer 分支输出与残差路径相加。

#ifndef KUIPER_INCLUDE_OP_ADD_H
#define KUIPER_INCLUDE_OP_ADD_H
#include "base/base.h"
#include "layer.h"
namespace op {
// VecAddLayer 做逐元素相加，主要用于 Transformer block 的残差连接。
class VecAddLayer : public Layer {
 public:
  explicit VecAddLayer(base::DeviceType device_type);

  base::Status check() const override;

  base::Status forward() override;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ADD_H
