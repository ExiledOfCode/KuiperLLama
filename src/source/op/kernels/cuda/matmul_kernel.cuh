// 文件说明：矩阵乘 kernel 实现，覆盖普通、cuBLAS、量化和实验优化路径。

#ifndef MATMUL_KERNEL_CU_CUH
#define MATMUL_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config = nullptr);

void matmul_kernel_cu_awq_int4(const tensor::Tensor& input, const tensor::Tensor& weight,
                               const tensor::Tensor& output, int32_t group_size,
                               const tensor::Tensor& scale, const tensor::Tensor& zeros,
                               const CudaConfig* config = nullptr);
}  // namespace kernel

#endif  // MATMUL_KERNEL_CU_CUH
