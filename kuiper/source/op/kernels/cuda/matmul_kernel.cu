#include <tensor/tensor.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
namespace {

enum class MatmulCudaImpl {
  kKuiper = 0,
  kCublas = 1,
  kLabCpAsync = 2,
};

const char* matmul_impl_name(MatmulCudaImpl impl) {
  switch (impl) {
    case MatmulCudaImpl::kLabCpAsync:
      return "lab_cp_async";
    case MatmulCudaImpl::kCublas:
      return "cublas";
    case MatmulCudaImpl::kKuiper:
    default:
      return "kuiper_cuda";
  }
}

MatmulCudaImpl resolve_matmul_impl() {
  const char* raw = std::getenv("KLLM_OP_MATMUL_IMPL");
  if (raw == nullptr || *raw == '\0') {
    return MatmulCudaImpl::kKuiper;
  }

  std::string value(raw);
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (value == "kuiper_cuda" || value == "kuiper" || value == "default" || value == "auto") {
    return MatmulCudaImpl::kKuiper;
  }
  if (value == "cublas") {
    return MatmulCudaImpl::kCublas;
  }
  if (value == "lab_cp_async_fp32" || value == "lab_cp_async" || value == "lab_gemm" ||
      value == "lab") {
    return MatmulCudaImpl::kLabCpAsync;
  }

  LOG(WARNING) << "Unknown KLLM_OP_MATMUL_IMPL=" << value
               << ", fallback to kuiper_cuda.";
  return MatmulCudaImpl::kKuiper;
}

MatmulCudaImpl selected_matmul_impl() {
  static const MatmulCudaImpl impl = resolve_matmul_impl();
  static const char* impl_name = matmul_impl_name(impl);
  static std::once_flag log_once;
  std::call_once(log_once, []() {
    LOG(INFO) << "Selected CUDA matmul implementation: " << impl_name;
  });
  return impl;
}

int32_t matmul_input_rows(const tensor::Tensor& input) {
  if (input.dims_size() <= 1) {
    return 1;
  }
  return input.get_dim(input.dims_size() - 1);
}

cublasHandle_t shared_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    const cublasStatus_t status = cublasCreate(&handle);
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cublasCreate failed";
  });
  return handle;
}

void configure_cublas_stream(cublasHandle_t handle, const CudaConfig* config) {
  cudaStream_t stream = config ? config->stream : nullptr;
  const cublasStatus_t status = cublasSetStream(handle, stream);
  CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cublasSetStream failed";
}

bool lab_cp_async_runtime_supported() {
  static int supported = -1;
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    int device = 0;
    cudaDeviceProp prop{};
    const cudaError_t current_status = cudaGetDevice(&device);
    const cudaError_t prop_status =
        current_status == cudaSuccess ? cudaGetDeviceProperties(&prop, device) : current_status;
    supported = (prop_status == cudaSuccess && prop.major >= 8) ? 1 : 0;
  });
  return supported == 1;
}

void log_lab_cp_async_fallback_once(const char* reason) {
  static std::mutex log_mutex;
  static std::set<std::string> reasons;
  std::lock_guard<std::mutex> lock(log_mutex);
  const std::string key(reason ? reason : "unknown");
  if (reasons.insert(key).second) {
    LOG(WARNING) << "lab_cp_async matmul fallback to kuiper_cuda: " << key;
  }
}

__device__ __forceinline__ uint32_t cp_async_cvta_to_shared_u32(const void* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
  (void)ptr;
  return 0u;
#endif
}

template <int BYTES>
__device__ __forceinline__ void cp_async_ca(void* shared_ptr, const void* global_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
               :
               : "r"(cp_async_cvta_to_shared_u32(shared_ptr)), "l"(global_ptr), "n"(BYTES));
#else
  (void)shared_ptr;
  (void)global_ptr;
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
#endif
}

}  // namespace

template <typename T>
__device__ inline float to_float_device(T value);

template <>
__device__ inline float to_float_device<float>(float value) {
  return value;
}

template <>
__device__ inline float to_float_device<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__global__ void fp32_to_bf16_kernel(const float* input, __nv_bfloat16* output, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = __float2bfloat16(input[idx]);
  }
}

template <int BLOCK_ROWS, int TILE_K>
__global__ void matmul_kernel_cu_lab_cp_async_fp32_vec(const float* input, const float* weight,
                                                       float* output, int input_dim,
                                                       int output_dim, float scale) {
  __shared__ alignas(16) float shared_weight[2][BLOCK_ROWS][TILE_K];
  __shared__ float shared_input[2][TILE_K];

  const int lane = threadIdx.x;
  const int row = blockIdx.x * BLOCK_ROWS + lane;
  const bool can_vector_load = (input_dim % 4) == 0;
  float acc = 0.0f;
  int current = 0;

  if (lane < TILE_K) {
    shared_input[current][lane] = (lane < input_dim) ? input[lane] : 0.0f;
  }

  if (row < output_dim && input_dim >= TILE_K && can_vector_load) {
    cp_async_ca<16>(&shared_weight[current][lane][0], weight + row * input_dim);
    cp_async_ca<16>(&shared_weight[current][lane][4], weight + row * input_dim + 4);
  } else {
    for (int k = 0; k < TILE_K; ++k) {
      const int index = k;
      shared_weight[current][lane][k] =
          (row < output_dim && index < input_dim) ? weight[row * input_dim + index] : 0.0f;
    }
  }
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();

  for (int k_base = 0; k_base < input_dim; k_base += TILE_K) {
    const int next_k = k_base + TILE_K;
    if (next_k < input_dim) {
      const int next = current ^ 1;
      if (lane < TILE_K) {
        shared_input[next][lane] =
            (next_k + lane < input_dim) ? input[next_k + lane] : 0.0f;
      }

      const int remaining = input_dim - next_k;
      if (row < output_dim && remaining >= TILE_K && can_vector_load) {
        cp_async_ca<16>(&shared_weight[next][lane][0], weight + row * input_dim + next_k);
        cp_async_ca<16>(&shared_weight[next][lane][4], weight + row * input_dim + next_k + 4);
      } else {
        for (int k = 0; k < TILE_K; ++k) {
          const int index = next_k + k;
          shared_weight[next][lane][k] =
              (row < output_dim && index < input_dim) ? weight[row * input_dim + index] : 0.0f;
        }
      }
      cp_async_commit_group();
    }

    if (row < output_dim) {
#pragma unroll
      for (int k = 0; k < TILE_K; ++k) {
        acc += shared_weight[current][lane][k] * shared_input[current][k];
      }
    }

    if (next_k < input_dim) {
      cp_async_wait_group<0>();
    }
    __syncthreads();
    current ^= 1;
  }

  if (row < output_dim) {
    output[row] = acc * scale;
  }
}

template <int BLOCK_ROWS, int TILE_K>
__global__ void matmul_kernel_cu_lab_cp_async_bf16_vec(const float* input,
                                                       const __nv_bfloat16* weight, float* output,
                                                       int input_dim, int output_dim,
                                                       float scale) {
  __shared__ alignas(16) __nv_bfloat16 shared_weight[2][BLOCK_ROWS][TILE_K];
  __shared__ float shared_input[2][TILE_K];

  const int lane = threadIdx.x;
  const int row = blockIdx.x * BLOCK_ROWS + lane;
  const bool can_async_load = (input_dim % TILE_K) == 0;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  float acc = 0.0f;
  int current = 0;

  if (lane < TILE_K) {
    shared_input[current][lane] = (lane < input_dim) ? input[lane] : 0.0f;
  }

  if (row < output_dim && input_dim >= TILE_K && can_async_load) {
    cp_async_ca<16>(&shared_weight[current][lane][0], weight + row * input_dim);
  } else {
    for (int k = 0; k < TILE_K; ++k) {
      const int index = k;
      shared_weight[current][lane][k] =
          (row < output_dim && index < input_dim) ? weight[row * input_dim + index] : zero;
    }
  }
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();

  for (int k_base = 0; k_base < input_dim; k_base += TILE_K) {
    const int next_k = k_base + TILE_K;
    if (next_k < input_dim) {
      const int next = current ^ 1;
      if (lane < TILE_K) {
        shared_input[next][lane] =
            (next_k + lane < input_dim) ? input[next_k + lane] : 0.0f;
      }

      const int remaining = input_dim - next_k;
      if (row < output_dim && remaining >= TILE_K && can_async_load) {
        cp_async_ca<16>(&shared_weight[next][lane][0], weight + row * input_dim + next_k);
      } else {
        for (int k = 0; k < TILE_K; ++k) {
          const int index = next_k + k;
          shared_weight[next][lane][k] =
              (row < output_dim && index < input_dim) ? weight[row * input_dim + index] : zero;
        }
      }
      cp_async_commit_group();
    }

    if (row < output_dim) {
#pragma unroll
      for (int k = 0; k < TILE_K; ++k) {
        acc += to_float_device(shared_weight[current][lane][k]) * shared_input[current][k];
      }
    }

    if (next_k < input_dim) {
      cp_async_wait_group<0>();
    }
    __syncthreads();
    current ^= 1;
  }

  if (row < output_dim) {
    output[row] = acc * scale;
  }
}

template <typename WeightT, int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_mixed(const float* input, const WeightT* weight, float* output,
                                       int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0.f;
    int row_offset = p * M;
    for (int i = tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * to_float_device(weight[row_offset + i]);
    }

    __syncthreads();
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
    for (int i = tid; i < pack_num; i += blockDim.x) {
      float4 input_float4 = *(input_float4_ptr + i);
      float4 weight_float4 = *(weight_float4_ptr + i);
      float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                       input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[row_offset + i];
    }

    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32_awq_int4(const float* input, const uint8_t* packed_weight,
                                               const float* scales, const uint8_t* zeros,
                                               const int32_t group_size, float* output, int M,
                                               int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    if (p >= K) {
      return;
    }
    sdata[tid] = 0.f;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const uint8_t packed = packed_weight[weight_idx >> 1];
      const uint8_t q = (weight_idx & 1) == 0 ? (packed & 0x0f) : (packed >> 4);
      const int group_idx = weight_idx / group_size;
      const float dequant =
          (static_cast<float>(q) - static_cast<float>(zeros[group_idx] & 0x0f)) *
          scales[group_idx];
      sdata[tid] += input[i] * dequant;
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu_kuiper(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, const float scale,
                             const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  CHECK_EQ(M, input.get_dim(0));
  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    const __nv_bfloat16* weight_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    if (config && config->stream) {
      matmul_kernel_cu_mixed<__nv_bfloat16, 128, 1><<<K, 128, 0, config->stream>>>(
          input.ptr<float>(), weight_ptr, const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_mixed<__nv_bfloat16, 128, 1><<<K, 128>>>(
          input.ptr<float>(), weight_ptr, const_cast<float*>(output.ptr<float>()), M, K);
    }
  } else {
    if (config && config->stream) {
      matmul_kernel_cu_fp32<128, 1><<<K, 128, 0, config->stream>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K);
    } else {
      matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                                const_cast<float*>(output.ptr<float>()), M, K);
    }
  }
}

void matmul_kernel_cu_cublas(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, const float scale,
                             const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.is_empty() == false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t output_dim = weight.get_dim(0);
  const int32_t input_dim = weight.get_dim(1);
  const int32_t rows = matmul_input_rows(input);
  CHECK_EQ(input.get_dim(0), input_dim);
  CHECK_EQ(static_cast<int32_t>(output.size()), output_dim * rows);

  cublasHandle_t handle = shared_cublas_handle();
  configure_cublas_stream(handle, config);

  const float alpha = scale;
  const float beta = 0.0f;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor input_bf16(base::DataType::kDataTypeBf16, input.dims(), true, alloc_cu);
    const int input_size = static_cast<int>(input.size());
    const int block_size = 256;
    const int grid_size = (input_size + block_size - 1) / block_size;
    if (config && config->stream) {
      fp32_to_bf16_kernel<<<grid_size, block_size, 0, config->stream>>>(
          input.ptr<float>(), reinterpret_cast<__nv_bfloat16*>(input_bf16.ptr<uint16_t>()),
          input_size);
    } else {
      fp32_to_bf16_kernel<<<grid_size, block_size>>>(
          input.ptr<float>(), reinterpret_cast<__nv_bfloat16*>(input_bf16.ptr<uint16_t>()),
          input_size);
    }
    status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, rows, input_dim, &alpha,
        weight.ptr<uint16_t>(), CUDA_R_16BF, input_dim, input_bf16.ptr<uint16_t>(), CUDA_R_16BF,
        input_dim, &beta, const_cast<float*>(output.ptr<float>()), CUDA_R_32F, output_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, rows, input_dim, &alpha,
                         weight.ptr<float>(), input_dim, input.ptr<float>(), input_dim, &beta,
                         const_cast<float*>(output.ptr<float>()), output_dim);
  }
  CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cuBLAS matmul failed";
}

void matmul_kernel_cu_lab_cp_async(const tensor::Tensor& input, const tensor::Tensor& weight,
                                   const tensor::Tensor& output, const float scale,
                                   const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.is_empty() == false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  if (!lab_cp_async_runtime_supported()) {
    log_lab_cp_async_fallback_once("requires sm80+ runtime support");
    matmul_kernel_cu_kuiper(input, weight, output, scale, config);
    return;
  }
  if (input.dims_size() != 1) {
    log_lab_cp_async_fallback_once("currently only supports 1D decode matmul");
    matmul_kernel_cu_kuiper(input, weight, output, scale, config);
    return;
  }
  if (weight.data_type() != base::DataType::kDataTypeFp32 &&
      weight.data_type() != base::DataType::kDataTypeBf16) {
    log_lab_cp_async_fallback_once("currently only supports fp32/bf16 weights");
    matmul_kernel_cu_kuiper(input, weight, output, scale, config);
    return;
  }

  const int32_t output_dim = weight.get_dim(0);
  const int32_t input_dim = weight.get_dim(1);
  CHECK_EQ(input.get_dim(0), input_dim);
  CHECK_EQ(static_cast<int32_t>(output.size()), output_dim);

  constexpr int kBlockRows = 128;
  constexpr int kTileK = 8;
  constexpr int kThreads = kBlockRows;
  const int grid_size = (output_dim + kBlockRows - 1) / kBlockRows;
  if (weight.data_type() == base::DataType::kDataTypeBf16) {
    const __nv_bfloat16* weight_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.ptr<uint16_t>());
    if (config && config->stream) {
      matmul_kernel_cu_lab_cp_async_bf16_vec<kBlockRows, kTileK>
          <<<grid_size, kThreads, 0, config->stream>>>(
              input.ptr<float>(), weight_ptr, const_cast<float*>(output.ptr<float>()), input_dim,
              output_dim, scale);
    } else {
      matmul_kernel_cu_lab_cp_async_bf16_vec<kBlockRows, kTileK><<<grid_size, kThreads>>>(
          input.ptr<float>(), weight_ptr, const_cast<float*>(output.ptr<float>()), input_dim,
          output_dim, scale);
    }
  } else {
    if (config && config->stream) {
      matmul_kernel_cu_lab_cp_async_fp32_vec<kBlockRows, kTileK>
          <<<grid_size, kThreads, 0, config->stream>>>(
              input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()),
              input_dim, output_dim, scale);
    } else {
      matmul_kernel_cu_lab_cp_async_fp32_vec<kBlockRows, kTileK><<<grid_size, kThreads>>>(
          input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()),
          input_dim, output_dim, scale);
    }
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  switch (selected_matmul_impl()) {
    case MatmulCudaImpl::kLabCpAsync:
      matmul_kernel_cu_lab_cp_async(input, weight, output, scale, config);
      return;
    case MatmulCudaImpl::kCublas:
      matmul_kernel_cu_cublas(input, weight, output, scale, config);
      return;
    case MatmulCudaImpl::kKuiper:
    default:
      matmul_kernel_cu_kuiper(input, weight, output, scale, config);
      return;
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                  scale.ptr<float>(), group_size,
                                                  const_cast<float*>(output.ptr<float>()), M, K);
  }
}

void matmul_kernel_cu_awq_int4(const tensor::Tensor& input, const tensor::Tensor& weight,
                               const tensor::Tensor& output, int32_t group_size,
                               const tensor::Tensor& scale, const tensor::Tensor& zeros,
                               const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 1);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(scale.is_empty() == false && scale.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(zeros.is_empty() == false && zeros.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.data_type() == base::DataType::kDataTypeInt8);
  CHECK(zeros.data_type() == base::DataType::kDataTypeInt8);

  const int32_t K = output.get_dim(0);
  const int32_t M = input.get_dim(0);
  CHECK_GT(group_size, 0);
  CHECK_EQ((static_cast<int64_t>(K) * M) % group_size, 0);
  CHECK_EQ(static_cast<int64_t>(weight.size()) * 2, static_cast<int64_t>(K) * M);
  CHECK_EQ(scale.size(), static_cast<size_t>(K) * static_cast<size_t>(M) / group_size);
  CHECK_EQ(zeros.size(), scale.size());

  if (config->stream) {
    matmul_kernel_cu_fp32_awq_int4<128, 1><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<uint8_t>(), scale.ptr<float>(), zeros.ptr<uint8_t>(),
        group_size, const_cast<float*>(output.ptr<float>()), M, K);
  } else {
    matmul_kernel_cu_fp32_awq_int4<128, 1><<<K, 128>>>(
        input.ptr<float>(), weight.ptr<uint8_t>(), scale.ptr<float>(), zeros.ptr<uint8_t>(),
        group_size, const_cast<float*>(output.ptr<float>()), M, K);
  }
}
}  // namespace kernel
