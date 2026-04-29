// 文件说明：Argmax 采样器声明，在温度为零或贪心模式下选择最大 logit。

//
// Created by fss on 24-6-9.
//

#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
namespace sampler {
// 贪心采样：返回 logits 中最大值的下标。用于 temperature=0 的确定性生成。
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

  size_t sample(const float* logits, size_t size, void* stream) override;
};
}  // namespace sampler
#endif  // LLAMA_INFER_NON_SAMPLER_H
