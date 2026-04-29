// 文件说明：矩阵乘 kernel 实现，覆盖普通、cuBLAS、量化和实验优化路径。

#ifndef LLAMA_INFER_MATMUL_KERNEL_H
#define LLAMA_INFER_MATMUL_KERNEL_H
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const CudaConfig* config = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_MATMUL_KERNEL_H
