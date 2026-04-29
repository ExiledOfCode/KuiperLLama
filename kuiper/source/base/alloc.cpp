// 文件说明：allocator 工厂实现，提供全局 CPU/CUDA 分配器实例。

#include "base/alloc.h"
#include <cuda_runtime_api.h>
namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size) {
    return;
  }

  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  // 这里不根据 allocator 自身的 device_type 推断方向，而是完全信任调用方传入的 memcpy_kind。
  // 这样同一个 allocator 可以处理 CPU<->CUDA、CUDA<->CUDA 等多种拷贝组合。
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
  if (need_sync) {
    // 初始化、测试和服务端清理阶段会要求同步，确保后续读取看到确定结果。
    cudaDeviceSynchronize();
  }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  // 清零用于每次请求前重置运行时 buffer；CPU 直接 memset，CUDA 支持 stream 异步清零。
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      cudaMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
}

}  // namespace base
