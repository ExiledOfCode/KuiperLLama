// 文件说明：温度采样 kernel 实现，根据概率分布随机选择下一个 token。

#ifndef SRC_SOURCE_OP_KERNELS_CUDA_TEMPERATURE_SAMPLER_KERNEL_CUH_
#define SRC_SOURCE_OP_KERNELS_CUDA_TEMPERATURE_SAMPLER_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace kernel {

size_t temperature_sample_kernel_cu(const float* logits, size_t size, float temperature,
                                    uint64_t seed, void* stream);

}  // namespace kernel

#endif  // SRC_SOURCE_OP_KERNELS_CUDA_TEMPERATURE_SAMPLER_KERNEL_CUH_
