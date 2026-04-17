#include "temperature_sampler_kernel.cuh"

#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include "base/alloc.h"

namespace kernel {
namespace {

__device__ uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

__device__ float random_uniform_01(uint64_t seed) {
  const uint64_t value = splitmix64(seed);
  const uint32_t mantissa = static_cast<uint32_t>((value >> 40U) & 0xFFFFFFU);
  return (static_cast<float>(mantissa) + 0.5f) / 16777216.0f;
}

__global__ void temperature_sample_kernel(const float* logits, size_t size, float temperature,
                                          uint64_t seed, size_t* output_idx) {
  __shared__ float shared_values[512];
  __shared__ size_t shared_indices[512];

  const uint32_t tid = threadIdx.x;
  float local_max = -FLT_MAX;
  size_t local_idx = SIZE_MAX;
  for (size_t i = tid; i < size; i += blockDim.x) {
    const float value = logits[i];
    if (value > local_max || (value == local_max && i < local_idx)) {
      local_max = value;
      local_idx = i;
    }
  }

  shared_values[tid] = local_max;
  shared_indices[tid] = local_idx;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1U; stride > 0U; stride >>= 1U) {
    if (tid < stride) {
      const float other_value = shared_values[tid + stride];
      const size_t other_idx = shared_indices[tid + stride];
      if (other_value > shared_values[tid] ||
          (other_value == shared_values[tid] && other_idx < shared_indices[tid])) {
        shared_values[tid] = other_value;
        shared_indices[tid] = other_idx;
      }
    }
    __syncthreads();
  }

  const float max_value = shared_values[0];
  const size_t max_idx = shared_indices[0];
  const float inv_temperature = 1.0f / temperature;
  float local_sum = 0.0f;
  for (size_t i = tid; i < size; i += blockDim.x) {
    local_sum += expf((logits[i] - max_value) * inv_temperature);
  }

  shared_values[tid] = local_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x >> 1U; stride > 0U; stride >>= 1U) {
    if (tid < stride) {
      shared_values[tid] += shared_values[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0U) {
    const float sum_value = shared_values[0];
    if (!isfinite(sum_value) || sum_value <= 0.0f) {
      *output_idx = max_idx;
      return;
    }

    const float target = random_uniform_01(seed) * sum_value;
    float cumulative = 0.0f;
    size_t selected = max_idx;
    for (size_t i = 0; i < size; ++i) {
      cumulative += expf((logits[i] - max_value) * inv_temperature);
      if (cumulative >= target) {
        selected = i;
        break;
      }
    }
    *output_idx = selected;
  }
}

}  // namespace

size_t temperature_sample_kernel_cu(const float* logits, size_t size, float temperature,
                                    uint64_t seed, void* stream) {
  if (logits == nullptr || size == 0) {
    return 0;
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();
  size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
  if (index == nullptr) {
    return 0;
  }

  size_t output_index = 0;
  if (!stream) {
    temperature_sample_kernel<<<1, 512>>>(logits, size, temperature, seed, index);
    cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    temperature_sample_kernel<<<1, 512, 0, stream_>>>(logits, size, temperature, seed, index);
    cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }
  alloc_cu->release(index);
  return output_index;
}

}  // namespace kernel
