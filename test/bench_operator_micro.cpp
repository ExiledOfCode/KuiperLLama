// 文件说明：算子微基准，直接测试矩阵乘法和 RMSNorm 不同实现版本的吞吐。

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/buffer.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
void matmul_kernel_cu_kuiper(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const CudaConfig* config);
void matmul_kernel_cu_cublas(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const CudaConfig* config);
void matmul_kernel_cu_lab_cp_async(const tensor::Tensor& input, const tensor::Tensor& weight,
                                   const tensor::Tensor& output, float scale,
                                   const CudaConfig* config);
void rmsnorm_kernel_cu_dim_kuiper(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t dim, void* stream);
void rmsnorm_kernel_cu_lab(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int rows, int cols, void* stream);
}  // namespace kernel

namespace {

struct MatmulCase {
  int input_dim;
  int output_dim;
};

struct RmsNormCase {
  int rows;
  int cols;
};

struct Version {
  std::string id;
  std::string label;
};

int int_arg(int argc, char** argv, const std::string& name, int fallback) {
  const std::string prefix = "--" + name + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind(prefix, 0) == 0) {
      return std::max(1, std::atoi(arg.substr(prefix.size()).c_str()));
    }
  }
  return fallback;
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    LOG(FATAL) << what << " failed: " << cudaGetErrorString(err);
  }
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  return ms;
}

void fill_tensor(tensor::Tensor& tensor, float base, float step) {
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.index<float>(i) = base + static_cast<float>((i % 257) - 128) * step;
  }
}

double bench_matmul_version(const std::string& version, const tensor::Tensor& input,
                            const tensor::Tensor& weight, const tensor::Tensor& output,
                            kernel::CudaConfig* config, int warmup, int iters) {
  auto run_once = [&]() {
    if (version == "original") {
      kernel::matmul_kernel_cu_kuiper(input, weight, output, 1.0f, config);
    } else if (version == "cublas") {
      kernel::matmul_kernel_cu_cublas(input, weight, output, 1.0f, config);
    } else if (version == "lab_cp_async") {
      kernel::matmul_kernel_cu_lab_cp_async(input, weight, output, 1.0f, config);
    } else {
      LOG(FATAL) << "unknown matmul version: " << version;
    }
  };

  for (int i = 0; i < warmup; ++i) {
    run_once();
  }
  check_cuda(cudaStreamSynchronize(config->stream), "warmup synchronize");

  cudaEvent_t start;
  cudaEvent_t stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  check_cuda(cudaEventRecord(start, config->stream), "cudaEventRecord(start)");
  for (int i = 0; i < iters; ++i) {
    run_once();
  }
  check_cuda(cudaEventRecord(stop, config->stream), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
  const double avg_ms = static_cast<double>(elapsed_ms(start, stop)) / static_cast<double>(iters);
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  return avg_ms;
}

double bench_rmsnorm_version(const std::string& version, const tensor::Tensor& input,
                             const tensor::Tensor& weight, const tensor::Tensor& output, int rows,
                             int cols, kernel::CudaConfig* config, int warmup, int iters) {
  auto run_once = [&]() {
    if (version == "original") {
      kernel::rmsnorm_kernel_cu_dim_kuiper(input, weight, output, 1, config->stream);
    } else if (version == "warp_reduce") {
      kernel::rmsnorm_kernel_cu_lab(input, weight, output, rows, cols, config->stream);
    } else {
      LOG(FATAL) << "unknown rmsnorm version: " << version;
    }
  };

  for (int i = 0; i < warmup; ++i) {
    run_once();
  }
  check_cuda(cudaStreamSynchronize(config->stream), "warmup synchronize");

  cudaEvent_t start;
  cudaEvent_t stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  check_cuda(cudaEventRecord(start, config->stream), "cudaEventRecord(start)");
  for (int i = 0; i < iters; ++i) {
    run_once();
  }
  check_cuda(cudaEventRecord(stop, config->stream), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
  const double avg_ms = static_cast<double>(elapsed_ms(start, stop)) / static_cast<double>(iters);
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  return avg_ms;
}

void print_header() {
  std::cout << "suite,version,label,input_dim,output_dim,rows,cols,avg_ms,gflops,gbps,iters\n";
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  const int warmup = int_arg(argc, argv, "warmup", 30);
  const int matmul_iters = int_arg(argc, argv, "matmul-iters", 200);
  const int rmsnorm_iters = int_arg(argc, argv, "rmsnorm-iters", 1000);

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  kernel::CudaConfig config;
  config.stream = stream;

  const std::vector<MatmulCase> matmul_cases = {
      {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}, {8192, 4096}};
  const std::vector<Version> matmul_versions = {
      {"original", "原版"}, {"cublas", "cuBLAS版本"}, {"lab_cp_async", "实验优化版本"}};

  const std::vector<RmsNormCase> rmsnorm_cases = {
      {64, 512}, {64, 1024}, {64, 2048}, {64, 4096}, {64, 8192}};
  const std::vector<Version> rmsnorm_versions = {
      {"original", "原版"}, {"warp_reduce", "Warp级归约版本"}};

  print_header();

  for (const auto& shape : matmul_cases) {
    tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, shape.input_dim, true, alloc_cpu);
    tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, shape.output_dim, shape.input_dim,
                              true, alloc_cpu);
    fill_tensor(input_cpu, 0.01f, 0.001f);
    fill_tensor(weight_cpu, -0.02f, 0.0005f);
    tensor::Tensor input = input_cpu.clone();
    tensor::Tensor weight = weight_cpu.clone();
    input.to_cuda(stream);
    weight.to_cuda(stream);
    tensor::Tensor output(base::DataType::kDataTypeFp32, shape.output_dim, true, alloc_cu);
    check_cuda(cudaStreamSynchronize(stream), "input copy synchronize");

    const double flops = 2.0 * static_cast<double>(shape.input_dim) *
                         static_cast<double>(shape.output_dim);
    for (const auto& version : matmul_versions) {
      const double avg_ms =
          bench_matmul_version(version.id, input, weight, output, &config, warmup, matmul_iters);
      const double gflops = flops / (avg_ms * 1.0e6);
      std::cout << "matmul," << version.id << "," << version.label << "," << shape.input_dim
                << "," << shape.output_dim << ",,,"
                << std::fixed << std::setprecision(6) << avg_ms << "," << gflops
                << ",," << matmul_iters << "\n";
    }
  }

  for (const auto& shape : rmsnorm_cases) {
    tensor::Tensor input_cpu(base::DataType::kDataTypeFp32, shape.rows, shape.cols, true,
                             alloc_cpu);
    tensor::Tensor weight_cpu(base::DataType::kDataTypeFp32, shape.cols, true, alloc_cpu);
    fill_tensor(input_cpu, 0.03f, 0.0007f);
    fill_tensor(weight_cpu, 0.95f, 0.0003f);
    tensor::Tensor input = input_cpu.clone();
    tensor::Tensor weight = weight_cpu.clone();
    input.to_cuda(stream);
    weight.to_cuda(stream);
    tensor::Tensor output(base::DataType::kDataTypeFp32, shape.rows, shape.cols, true, alloc_cu);
    check_cuda(cudaStreamSynchronize(stream), "input copy synchronize");

    // Approximate bytes: read input + read weight + write output.
    const double bytes = static_cast<double>(shape.rows) * static_cast<double>(shape.cols) *
                             sizeof(float) * 2.0 +
                         static_cast<double>(shape.cols) * sizeof(float);
    for (const auto& version : rmsnorm_versions) {
      const double avg_ms =
          bench_rmsnorm_version(version.id, input, weight, output, shape.rows, shape.cols, &config,
                                warmup, rmsnorm_iters);
      const double gbps = bytes / (avg_ms * 1.0e6);
      std::cout << "rmsnorm," << version.id << "," << version.label << ",,,"
                << shape.rows << "," << shape.cols << ","
                << std::fixed << std::setprecision(6) << avg_ms << ",," << gbps << ","
                << rmsnorm_iters << "\n";
    }
  }

  std::cout << std::flush;
  check_cuda(cudaStreamSynchronize(stream), "final cudaStreamSynchronize");
  // 该可执行文件只用于一次性微基准输出。这里直接退出，避免测试进程结束阶段
  // 触发已有 CUDA 全局资源析构顺序问题，GPU 资源会随进程退出释放。
  std::_Exit(0);
  return 0;
}
