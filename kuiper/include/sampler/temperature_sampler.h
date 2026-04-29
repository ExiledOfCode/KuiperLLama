// 文件说明：温度采样器声明，根据 temperature 在 logits 上执行随机采样。

#ifndef LLAMA_INFER_TEMPERATURE_SAMPLER_H
#define LLAMA_INFER_TEMPERATURE_SAMPLER_H

#include <cstdint>
#include "argmax_sampler.h"

namespace sampler {

// temperature <= 0 时退化为 ArgmaxSampler；temperature > 0 且 CUDA 可用时执行随机采样。
// 当前 CPU 路径保持贪心，避免在主机端额外维护 softmax/随机采样实现。
class TemperatureSampler : public Sampler {
 public:
  explicit TemperatureSampler(base::DeviceType device_type, float temperature = 0.0f);

  void set_temperature(float temperature);

  float temperature() const;

  size_t sample(const float* logits, size_t size, void* stream) override;

 private:
  ArgmaxSampler argmax_sampler_;
  float temperature_ = 0.0f;  // 0 表示贪心，值越大分布越平。
  uint64_t seed_ = 0;         // 启动时生成的基础随机种子。
  uint64_t sample_count_ = 0; // 每次采样扰动 seed，避免连续 token 使用同一随机序列。
};

}  // namespace sampler

#endif  // LLAMA_INFER_TEMPERATURE_SAMPLER_H
