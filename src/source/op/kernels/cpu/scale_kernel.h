// 文件说明：缩放 kernel 实现，对张量元素乘以标量系数。

#ifndef SCALE_KERNEL_H
#define SCALE_KERNEL_H
#include <tensor/tensor.h>
namespace kernel {
void scale_inplace_cpu(float scale, const tensor::Tensor& tensor, void* stream = nullptr);
}
#endif  // SCALE_KERNEL_H
