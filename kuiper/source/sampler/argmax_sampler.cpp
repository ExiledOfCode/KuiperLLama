// 文件说明：Argmax 采样器实现，按设备选择 CPU 或 CUDA 的最大 logit 查找。

#include "sampler/argmax_sampler.h"
#include <algorithm>
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    // CPU 直接在线性 logits 数组中找最大值下标。
    size_t next = std::distance(logits, std::max_element(logits, logits + size));
    return next;
  } else {
    // CUDA logits 位于显存，必须交给设备端 kernel 归约。
    size_t next = kernel::argmax_kernel_cu(logits, size, stream);
    return next;
  }
}
}  // namespace sampler
