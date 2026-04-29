// 文件说明：单元测试文件，验证 test_cu_matmul 相关模块的正确性。

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
using namespace kernel;

namespace kernel {
void matmul_kernel_cu_cublas(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, const CudaConfig* config);
void matmul_kernel_cu_lab_cp_async(const tensor::Tensor& input, const tensor::Tensor& weight,
                                   const tensor::Tensor& output, float scale,
                                   const CudaConfig* config);
}

namespace {

uint16_t float_to_bf16_bits(float value) {
  const __nv_bfloat16 bf16 = __float2bfloat16(value);
  uint16_t bits = 0;
  std::memcpy(&bits, &bf16, sizeof(bits));
  return bits;
}

float bf16_bits_to_float(uint16_t bits) {
  __nv_bfloat16 bf16;
  std::memcpy(&bf16, &bits, sizeof(bits));
  return __bfloat162float(bf16);
}

}  // namespace

TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = float(i);
  }

  for (int i = 0; i < 16; ++i) {
    weight.index<float>(i) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 4, true, alloc_cu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          config);

  out_cu.to_cpu();
  for (int i = 0; i < out_cu.size(); ++i) {
    ASSERT_EQ(out_cu.index<float>(i), out_cpu.index<float>(i));
  }
}

TEST(test_matmul_cu, matmul_linear_course) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          nullptr);

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_linear_course_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

  input.index<float>(0) = float(1);
  input.index<float>(1) = float(1);
  input.index<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.index<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 3, true, alloc_cu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_cu.clone();
  out_cpu.to_cpu();

  ASSERT_EQ(out_cpu.index<float>(0), 0);
  ASSERT_EQ(out_cpu.index<float>(1), 3);
  ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_cublas_fp32_matches_cpu) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 5, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 5, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, 4, true, alloc_cu);
  tensor::Tensor golden(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  const float input_values[5] = {1.0f, -0.5f, 2.0f, 0.25f, -1.5f};
  const float weight_values[20] = {
      0.5f, 1.0f, -0.5f, 2.0f, 0.25f,    //
      -1.0f, 0.75f, 1.5f, -0.25f, 0.5f,  //
      0.2f, -0.8f, 0.3f, 1.2f, -1.1f,    //
      2.0f, -0.5f, 0.4f, -0.3f, 0.9f};

  for (int i = 0; i < 5; ++i) {
    input.index<float>(i) = input_values[i];
  }
  for (int i = 0; i < 20; ++i) {
    weight.index<float>(i) = weight_values[i];
  }

  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();
  input.to_cuda();
  weight.to_cuda();

  kernel::matmul_kernel_cu_cublas(input, weight, output, 1.f, nullptr);
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, golden, 1.f,
                                                          nullptr);

  tensor::Tensor output_cpu = output.clone();
  output_cpu.to_cpu();
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(output_cpu.index<float>(i), golden.index<float>(i), 1e-5f);
  }
}

TEST(test_matmul_cu, matmul_cublas_bf16_matches_manual_reference) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 7, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeBf16, 3, 7, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, 3, true, alloc_cu);
  uint16_t weight_bits[21] = {};

  const float input_values[7] = {0.25f, -1.0f, 1.5f, 2.0f, -0.75f, 0.4f, 1.25f};
  const float weight_values[21] = {
      0.5f, -1.25f, 0.75f, 0.3f, -0.6f, 1.2f, -0.4f,   //
      -0.9f, 0.2f, 1.1f, -1.4f, 0.8f, 0.5f, -0.3f,     //
      1.6f, -0.7f, 0.45f, 0.9f, -1.1f, 0.35f, 0.65f};

  for (int i = 0; i < 7; ++i) {
    input.index<float>(i) = input_values[i];
  }
  for (int i = 0; i < 21; ++i) {
    weight_bits[i] = float_to_bf16_bits(weight_values[i]);
    weight.index<uint16_t>(i) = weight_bits[i];
  }

  input.to_cuda();
  weight.to_cuda();
  kernel::matmul_kernel_cu_cublas(input, weight, output, 1.f, nullptr);

  tensor::Tensor output_cpu = output.clone();
  output_cpu.to_cpu();
  for (int row = 0; row < 3; ++row) {
    float expected = 0.0f;
    for (int col = 0; col < 7; ++col) {
      const float weight_value = bf16_bits_to_float(weight_bits[row * 7 + col]);
      expected += input_values[col] * weight_value;
    }
    ASSERT_NEAR(output_cpu.index<float>(row), expected, 2e-2f);
  }
}

TEST(test_matmul_cu, matmul_lab_cp_async_fp32_matches_cpu_large_shape) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  constexpr int input_dim = 136;
  constexpr int output_dim = 257;
  tensor::Tensor input(base::DataType::kDataTypeFp32, input_dim, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, output_dim, input_dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, output_dim, true, alloc_cu);
  tensor::Tensor golden(base::DataType::kDataTypeFp32, output_dim, true, alloc_cpu);

  for (int i = 0; i < input_dim; ++i) {
    input.index<float>(i) = static_cast<float>((i % 17) * 0.125f - 0.75f);
  }
  for (int i = 0; i < output_dim * input_dim; ++i) {
    weight.index<float>(i) = static_cast<float>((i % 23) * 0.03125f - 0.3f);
  }

  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();
  input.to_cuda();
  weight.to_cuda();

  kernel::matmul_kernel_cu_lab_cp_async(input, weight, output, 0.5f, nullptr);
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, golden, 0.5f,
                                                          nullptr);

  tensor::Tensor output_cpu = output.clone();
  output_cpu.to_cpu();
  for (int i = 0; i < output_dim; ++i) {
    ASSERT_NEAR(output_cpu.index<float>(i), golden.index<float>(i), 1e-4f);
  }
}

TEST(test_matmul_cu, matmul_lab_cp_async_bf16_matches_reference_aligned_shape) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  constexpr int input_dim = 136;
  constexpr int output_dim = 257;
  tensor::Tensor input(base::DataType::kDataTypeFp32, input_dim, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeBf16, output_dim, input_dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, output_dim, true, alloc_cu);
  float input_values[input_dim] = {};
  uint16_t weight_bits[output_dim * input_dim] = {};

  for (int i = 0; i < input_dim; ++i) {
    input_values[i] = static_cast<float>((i % 17) * 0.125f - 0.75f);
    input.index<float>(i) = input_values[i];
  }
  for (int i = 0; i < output_dim * input_dim; ++i) {
    weight_bits[i] = float_to_bf16_bits(static_cast<float>((i % 23) * 0.03125f - 0.3f));
    weight.index<uint16_t>(i) = weight_bits[i];
  }

  input.to_cuda();
  weight.to_cuda();
  kernel::matmul_kernel_cu_lab_cp_async(input, weight, output, 0.75f, nullptr);

  tensor::Tensor output_cpu = output.clone();
  output_cpu.to_cpu();
  for (int row = 0; row < output_dim; ++row) {
    float expected = 0.0f;
    for (int col = 0; col < input_dim; ++col) {
      expected += input_values[col] * bf16_bits_to_float(weight_bits[row * input_dim + col]);
    }
    ASSERT_NEAR(output_cpu.index<float>(row), expected * 0.75f, 2e-3f);
  }
}

TEST(test_matmul_cu, matmul_lab_cp_async_bf16_matches_reference_unaligned_shape) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  constexpr int input_dim = 33;
  constexpr int output_dim = 19;
  tensor::Tensor input(base::DataType::kDataTypeFp32, input_dim, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeBf16, output_dim, input_dim, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, output_dim, true, alloc_cu);
  float input_values[input_dim] = {};
  uint16_t weight_bits[output_dim * input_dim] = {};

  for (int i = 0; i < input_dim; ++i) {
    input_values[i] = static_cast<float>((i % 9) * 0.2f - 0.5f);
    input.index<float>(i) = input_values[i];
  }
  for (int i = 0; i < output_dim * input_dim; ++i) {
    weight_bits[i] = float_to_bf16_bits(static_cast<float>((i % 11) * 0.04f - 0.2f));
    weight.index<uint16_t>(i) = weight_bits[i];
  }

  input.to_cuda();
  weight.to_cuda();
  kernel::matmul_kernel_cu_lab_cp_async(input, weight, output, 1.f, nullptr);

  tensor::Tensor output_cpu = output.clone();
  output_cpu.to_cpu();
  for (int row = 0; row < output_dim; ++row) {
    float expected = 0.0f;
    for (int col = 0; col < input_dim; ++col) {
      expected += input_values[col] * bf16_bits_to_float(weight_bits[row * input_dim + col]);
    }
    ASSERT_NEAR(output_cpu.index<float>(row), expected, 2e-3f);
  }
}
