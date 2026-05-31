// 文件说明：采样器抽象基类，统一 CPU/CUDA 采样实现接口。

#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
namespace sampler {
// 采样器只消费 logits 指针和 vocab size，不拥有 logits 内存。
// stream 仅 CUDA 路径使用，CPU 路径会忽略它。
class Sampler {
 public:
  explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

  virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

 protected:
  base::DeviceType device_type_;
};
}  // namespace sampler
#endif  // LLAMA_INFER_SAMPLER_H
