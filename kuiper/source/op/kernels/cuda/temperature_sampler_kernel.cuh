#ifndef KUIPER_SOURCE_OP_KERNELS_CUDA_TEMPERATURE_SAMPLER_KERNEL_CUH_
#define KUIPER_SOURCE_OP_KERNELS_CUDA_TEMPERATURE_SAMPLER_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace kernel {

size_t temperature_sample_kernel_cu(const float* logits, size_t size, float temperature,
                                    uint64_t seed, void* stream);

}  // namespace kernel

#endif  // KUIPER_SOURCE_OP_KERNELS_CUDA_TEMPERATURE_SAMPLER_KERNEL_CUH_
