// 文件说明：温度采样器实现，温度为零时退化为 argmax，否则执行概率采样。

#include "sampler/temperature_sampler.h"

#include <chrono>
#include <cmath>
#include "../op/kernels/cuda/temperature_sampler_kernel.cuh"

namespace sampler {

TemperatureSampler::TemperatureSampler(base::DeviceType device_type, float temperature)
    : Sampler(device_type), argmax_sampler_(device_type) {
  set_temperature(temperature);
  // 使用时间戳初始化基础 seed；真正传入 kernel 的 seed 会再混入 sample_count_。
  seed_ = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

void TemperatureSampler::set_temperature(float temperature) {
  if (!std::isfinite(temperature) || temperature < 0.0f) {
    temperature_ = 0.0f;
    return;
  }
  temperature_ = temperature;
}

float TemperatureSampler::temperature() const { return temperature_; }

size_t TemperatureSampler::sample(const float* logits, size_t size, void* stream) {
  if (temperature_ <= 0.0f || device_type_ != base::DeviceType::kDeviceCUDA) {
    // CPU 暂不实现随机采样；非 CUDA 或 temperature=0 统一走确定性 argmax。
    return argmax_sampler_.sample(logits, size, stream);
  }

  sample_count_ += 1;
  // 黄金比例常数用于混合计数，降低相邻 token seed 的相关性。
  const uint64_t seed = seed_ ^ (sample_count_ * 0x9E3779B97F4A7C15ULL);
  return kernel::temperature_sample_kernel_cu(logits, size, temperature_, seed, stream);
}

}  // namespace sampler
