// 文件说明：缩放 kernel 实现，对张量元素乘以标量系数。

#include "scale_kernel.h"
namespace kernel {
void scale_inplace_cpu(float scale, const tensor::Tensor& tensor, void* stream) {
  UNUSED(stream);
  CHECK(tensor.is_empty() == false);
  arma::fvec tensor_mat(const_cast<float*>(tensor.ptr<float>()), tensor.size(), false,
                        true);
  tensor_mat = tensor_mat * scale;
}
}  // namespace kernel