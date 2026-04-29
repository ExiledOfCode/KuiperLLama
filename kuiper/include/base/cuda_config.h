// 文件说明：CUDA 流与 cuBLAS 句柄配置对象，统一算子执行时的 GPU 上下文。

#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
