#ifndef LLAMA_INFER_TEMPERATURE_SAMPLER_H
#define LLAMA_INFER_TEMPERATURE_SAMPLER_H

#include <cstdint>
#include "argmax_sampler.h"

namespace sampler {

class TemperatureSampler : public Sampler {
 public:
  explicit TemperatureSampler(base::DeviceType device_type, float temperature = 0.0f);

  void set_temperature(float temperature);

  float temperature() const;

  size_t sample(const float* logits, size_t size, void* stream) override;

 private:
  ArgmaxSampler argmax_sampler_;
  float temperature_ = 0.0f;
  uint64_t seed_ = 0;
  uint64_t sample_count_ = 0;
};

}  // namespace sampler

#endif  // LLAMA_INFER_TEMPERATURE_SAMPLER_H
