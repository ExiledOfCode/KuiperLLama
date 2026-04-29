// 文件说明：SwiGLU 激活层声明，组合门控分支和上投影分支。

#ifndef LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
#define LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
#include "layer.h"
namespace op {
// SwiGLULayer 计算 silu(input1) * input2，通常 input1 来自 w1/gate，input2 来自 w3/up。
class SwiGLULayer : public op::Layer {
 public:
  explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t hidden_dim_ = 0;
};
}  // namespace op
#endif  // LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
