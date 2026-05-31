// 文件说明：RMSNorm kernel 实现，对隐藏向量执行均方根归一化。

#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {
namespace {

enum class RMSNormCudaImpl {
  kSrc = 0,
  kLabWarpReduce = 1,
};

const char* rmsnorm_impl_name(RMSNormCudaImpl impl) {
  switch (impl) {
    case RMSNormCudaImpl::kLabWarpReduce:
      return "lab_warp_reduce";
    case RMSNormCudaImpl::kSrc:
    default:
      return "src_cuda";
  }
}

RMSNormCudaImpl resolve_rmsnorm_impl() {
  const char* raw = std::getenv("KLLM_OP_RMSNORM_IMPL");
  if (raw == nullptr || *raw == '\0') {
    return RMSNormCudaImpl::kSrc;
  }

  std::string value(raw);
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (value == "src_cuda" || value == "src" || value == "default" || value == "auto") {
    return RMSNormCudaImpl::kSrc;
  }
  if (value == "lab_warp_reduce" || value == "lab") {
    return RMSNormCudaImpl::kLabWarpReduce;
  }

  LOG(WARNING) << "Unknown KLLM_OP_RMSNORM_IMPL=" << value
               << ", fallback to src_cuda.";
  return RMSNormCudaImpl::kSrc;
}

RMSNormCudaImpl selected_rmsnorm_impl() {
  static const RMSNormCudaImpl impl = resolve_rmsnorm_impl();
  static const char* impl_name = rmsnorm_impl_name(impl);
  static std::once_flag log_once;
  std::call_once(log_once, []() {
    LOG(INFO) << "Selected CUDA RMSNorm implementation: " << impl_name;
  });
  return impl;
}

template <typename T>
__device__ inline float lab_weight_to_float(T value);

template <>
__device__ inline float lab_weight_to_float<float>(float value) {
  return value;
}

template <>
__device__ inline float lab_weight_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename WeightT>
const WeightT* rmsnorm_weight_ptr(const tensor::Tensor& weight);

template <>
const float* rmsnorm_weight_ptr<float>(const tensor::Tensor& weight) {
  return weight.ptr<float>();
}

template <>
const __nv_bfloat16* rmsnorm_weight_ptr<__nv_bfloat16>(const tensor::Tensor& weight) {
  return reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
}

__device__ inline float warp_reduce_sum_lab(float value, unsigned mask = 0xffffffffu) {
  value += __shfl_down_sync(mask, value, 16);
  value += __shfl_down_sync(mask, value, 8);
  value += __shfl_down_sync(mask, value, 4);
  value += __shfl_down_sync(mask, value, 2);
  value += __shfl_down_sync(mask, value, 1);
  return value;
}

template <typename WeightT>
__global__ void row_rmsnorm_lab_kernel(const float* in, const WeightT* wei, float* out, int rows,
                                       int cols, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_count = (blockDim.x + 31) >> 5;
  if (row >= rows) {
    return;
  }

  __shared__ float warp_sums[32];
  __shared__ float inv_rms;

  const float* row_in = in + static_cast<size_t>(row) * cols;
  float* row_out = out + static_cast<size_t>(row) * cols;

  float sum = 0.0f;
  for (int idx = tid; idx < cols; idx += blockDim.x) {
    const float value = row_in[idx];
    sum += value * value;
  }

  sum = warp_reduce_sum_lab(sum);
  if (lane == 0) {
    warp_sums[wid] = sum;
  }
  __syncthreads();

  if (wid == 0) {
    float block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum_lab(block_sum);
    if (lane == 0) {
      inv_rms = rsqrtf(block_sum / static_cast<float>(cols) + eps);
    }
  }
  __syncthreads();

  for (int idx = tid; idx < cols; idx += blockDim.x) {
    row_out[idx] = lab_weight_to_float(wei[idx]) * row_in[idx] * inv_rms;
  }
}

template <typename WeightT>
void launch_rmsnorm_lab_kernel(const tensor::Tensor& input, const tensor::Tensor& weight,
                               const tensor::Tensor& output, int rows, int cols, float eps,
                               void* stream) {
  constexpr int kThreads = 256;
  const WeightT* weight_ptr = rmsnorm_weight_ptr<WeightT>(weight);
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_lab_kernel<<<rows, kThreads, 0, stream_>>>(
        input.ptr<float>(), weight_ptr, const_cast<float*>(output.ptr<float>()), rows, cols, eps);
  } else {
    row_rmsnorm_lab_kernel<<<rows, kThreads>>>(
        input.ptr<float>(), weight_ptr, const_cast<float*>(output.ptr<float>()), rows, cols, eps);
  }
}

}  // namespace

template <typename T>
__device__ inline float rmsnorm_to_float(T value);

template <>
__device__ inline float rmsnorm_to_float<float>(float value) {
  return value;
}

template <>
__device__ inline float rmsnorm_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size,
                                           int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

template <typename WeightT>
static __global__ void row_rmsnorm_f32_dim_mixed(float* in, const WeightT* wei, float* out,
                                                 int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  for (int i = tid; i < size; i += blockDim.x) {
    block_out[i] = rmsnorm_to_float(wei[i]) * block_in[i] * scale;
  }
}

template <typename WeightT, int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32_mixed(float* in, const WeightT* wei, float* out, int size,
                                             float eps) {
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  for (int i = tid; i < size; i += blockDim.x) {
    out[i] = rmsnorm_to_float(wei[i]) * in[i] * scale;
  }
}

void rmsnorm_kernel_cu_src(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    const __nv_bfloat16* wei_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_mixed<__nv_bfloat16, 128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr,
                                                                                 out_ptr, size, eps);
    } else {
      row_rmsnorm_f32_mixed<__nv_bfloat16, 128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size,
                                                                    eps);
    }
  } else {
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  }
}

void rmsnorm_kernel_cu_dim_src(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t dim, void* stream) {
  UNUSED(dim);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    const __nv_bfloat16* wei_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_dim_mixed<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                                        dim_size, size, eps);
    } else {
      row_rmsnorm_f32_dim_mixed<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                           size, eps);
    }
  } else {
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                                 dim_size, size, eps);
    } else {
      row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size,
                                                     eps);
    }
  }
}

void rmsnorm_kernel_cu_lab(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int rows, int cols, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif

  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    launch_rmsnorm_lab_kernel<__nv_bfloat16>(input, weight, output, rows, cols, eps, stream);
  } else {
    launch_rmsnorm_lab_kernel<float>(input, weight, output, rows, cols, eps, stream);
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  switch (selected_rmsnorm_impl()) {
    case RMSNormCudaImpl::kLabWarpReduce:
      rmsnorm_kernel_cu_lab(input, weight, output, 1, static_cast<int>(input.size()), stream);
      return;
    case RMSNormCudaImpl::kSrc:
    default:
      rmsnorm_kernel_cu_src(input, weight, output, stream);
      return;
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  const int total_size = static_cast<int>(input.size());
  const int row_size = input.get_dim(input.dims_size() - 1);
  const int rows = row_size > 0 ? total_size / row_size : 0;
  switch (selected_rmsnorm_impl()) {
    case RMSNormCudaImpl::kLabWarpReduce:
      rmsnorm_kernel_cu_lab(input, weight, output, rows, row_size, stream);
      return;
    case RMSNormCudaImpl::kSrc:
    default:
      rmsnorm_kernel_cu_dim_src(input, weight, output, dim, stream);
      return;
  }
}
}  // namespace kernel
