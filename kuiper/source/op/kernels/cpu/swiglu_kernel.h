// 文件说明：SwiGLU kernel 实现，融合 SiLU 激活和门控乘法。

#ifndef LLAMA_INFER_SWIGLU_KERNEL_H
#define LLAMA_INFER_SWIGLU_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream);
}  // namespace kernel
#endif  // LLAMA_INFER_SWIGLU_KERNEL_H
