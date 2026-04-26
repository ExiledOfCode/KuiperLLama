#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"

namespace kernel {
void rmsnorm_kernel_cu_lab(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int rows, int cols, void* stream);
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

float rmsnorm_expected_scale(const std::vector<float>& input_row, float eps) {
  float sum = 0.0f;
  for (float value : input_row) {
    sum += value * value;
  }
  return 1.0f / std::sqrt(sum / static_cast<float>(input_row.size()) + eps);
}

}  // namespace

TEST(test_rmsnorm_cu, rmsnorm_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }
  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, nullptr);
  out_cu.to_cpu();
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
}

TEST(test_rmsnorm_cu_dim, rmsnorm_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int dim_size = 4;
  int size = 1024;
  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, dim_size, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, dim_size, size, true, alloc_cpu);

  for (int i = 0; i < dim_size; ++i) {
    for (int j = 0; j < size; ++j) {
      wei_cpu.index<float>(j) = float(j);
      in_cpu.index<float>(i * size + j) = float(j);
    }
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, 1, nullptr);
  kernel::get_rmsnorm_dim_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, in_cu, 1, nullptr);

  out_cu.to_cpu();
  in_cu.to_cpu();

  tensor::Tensor in_cpu_golden(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu_golden(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_golden(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess);

  for (int j = 0; j < size; ++j) {
    wei_cpu_golden.index<float>(j) = float(j);
    in_cpu_golden.index<float>(j) = float(j);
  }
  tensor::Tensor in_cu_golden = in_cpu_golden.clone();
  tensor::Tensor wei_cu_golden = wei_cpu_golden.clone();
  tensor::Tensor out_cu_golden = out_cpu.clone();
  in_cu_golden.to_cuda();
  wei_cu_golden.to_cuda();
  out_cu_golden.to_cuda();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu_golden, wei_cu_golden,
                                                            out_cu_golden, nullptr);

  out_cu_golden.to_cpu();

  for (int i = 0; i < dim_size; ++i) {
    for (int j = 0; j < size; ++j) {
      ASSERT_EQ(out_cu.index<float>(i * size + j), out_cu_golden.index<float>(j))
          << "i: " << i << " j: " << j;
      ASSERT_EQ(in_cu.index<float>(i * size + j), out_cu_golden.index<float>(j))
          << "i: " << i << " j: " << j;
    }
  }
}

TEST(test_rmsnorm_cu, rmsnorm_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}

TEST(test_rmsnorm_cu, rmsnorm_stream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}

TEST(test_rmsnorm_cu, rmsnorm_lab_fp32_matches_manual_reference) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int rows = 3;
  const int cols = 65;
  tensor::Tensor input(base::DataType::kDataTypeFp32, rows, cols, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, cols, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, rows, cols, true, alloc_cu);
  std::vector<float> input_values(rows * cols, 0.0f);
  std::vector<float> weight_values(cols, 0.0f);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const float value = static_cast<float>((row + 1) * 0.25f + (col % 11) * 0.1f - 0.35f);
      input_values[row * cols + col] = value;
      input.index<float>(row * cols + col) = value;
    }
  }
  for (int col = 0; col < cols; ++col) {
    const float value = static_cast<float>(0.2f + (col % 7) * 0.05f);
    weight_values[col] = value;
    weight.index<float>(col) = value;
  }

  input.to_cuda();
  weight.to_cuda();
  kernel::rmsnorm_kernel_cu_lab(input, weight, output, rows, cols, nullptr);

  tensor::Tensor output_cpu = output.clone();
  output_cpu.to_cpu();

  const float eps = 1e-6f;
  for (int row = 0; row < rows; ++row) {
    std::vector<float> row_values;
    row_values.reserve(cols);
    for (int col = 0; col < cols; ++col) {
      row_values.push_back(input_values[row * cols + col]);
    }
    const float scale = rmsnorm_expected_scale(row_values, eps);
    for (int col = 0; col < cols; ++col) {
      const float expected = row_values[col] * weight_values[col] * scale;
      ASSERT_NEAR(output_cpu.index<float>(row * cols + col), expected, 1e-5f);
    }
  }
}

TEST(test_rmsnorm_cu, rmsnorm_lab_bf16_matches_manual_reference) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int rows = 2;
  const int cols = 96;
  tensor::Tensor input(base::DataType::kDataTypeFp32, rows, cols, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeBf16, cols, true, alloc_cpu);
  tensor::Tensor output(base::DataType::kDataTypeFp32, rows, cols, true, alloc_cu);
  std::vector<float> input_values(rows * cols, 0.0f);
  std::vector<uint16_t> weight_bits(cols, 0);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const float value = static_cast<float>((row + 1) * 0.3f - (col % 9) * 0.07f + 0.12f);
      input_values[row * cols + col] = value;
      input.index<float>(row * cols + col) = value;
    }
  }
  for (int col = 0; col < cols; ++col) {
    const float weight_value = static_cast<float>(0.15f + (col % 13) * 0.03f);
    weight_bits[col] = float_to_bf16_bits(weight_value);
    weight.index<uint16_t>(col) = weight_bits[col];
  }

  input.to_cuda();
  weight.to_cuda();
  kernel::rmsnorm_kernel_cu_lab(input, weight, output, rows, cols, nullptr);

  tensor::Tensor output_cpu = output.clone();
  output_cpu.to_cpu();

  const float eps = 1e-6f;
  for (int row = 0; row < rows; ++row) {
    std::vector<float> row_values;
    row_values.reserve(cols);
    for (int col = 0; col < cols; ++col) {
      row_values.push_back(input_values[row * cols + col]);
    }
    const float scale = rmsnorm_expected_scale(row_values, eps);
    for (int col = 0; col < cols; ++col) {
      const float weight_value = bf16_bits_to_float(weight_bits[col]);
      const float expected = row_values[col] * weight_value * scale;
      ASSERT_NEAR(output_cpu.index<float>(row * cols + col), expected, 2e-3f);
    }
  }
}
