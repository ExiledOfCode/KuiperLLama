// 文件说明：Softmax kernel 实现，提供注意力分数归一化能力。

#ifndef LLAMA_INFER_SOFTMAX_KERNEL_H
#define LLAMA_INFER_SOFTMAX_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_SOFTMAX_KERNEL_H
