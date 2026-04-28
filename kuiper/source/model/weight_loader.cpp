#include "model/weight_loader.h"

#include <algorithm>
#include <cuda_runtime_api.h>

#include <memory>
#include <vector>

#include <glog/logging.h>

#include "op/matmul.h"

namespace model {
namespace {

void append_tensor_if_valid(std::vector<tensor::Tensor*>& tensors, tensor::Tensor& tensor) {
  if (!tensor.is_empty()) {
    tensors.push_back(&tensor);
  }
}

void collect_layer_tensors(const std::shared_ptr<op::Layer>& layer,
                           std::vector<tensor::Tensor*>& tensors) {
  auto param_layer = std::dynamic_pointer_cast<op::LayerParam>(layer);
  if (!param_layer) {
    return;
  }

  for (size_t idx = 0; idx < param_layer->weight_size(); ++idx) {
    append_tensor_if_valid(tensors, param_layer->get_weight(static_cast<int32_t>(idx)));
  }
  if (param_layer->has_scales()) {
    append_tensor_if_valid(tensors, param_layer->get_scales());
  }
  if (param_layer->has_zeros()) {
    append_tensor_if_valid(tensors, param_layer->get_zeros());
  }

  auto matmul_layer = std::dynamic_pointer_cast<op::MatmulLayer>(layer);
  if (!matmul_layer || !matmul_layer->has_bias()) {
    return;
  }
  for (size_t idx = 0; idx < matmul_layer->bias_size(); ++idx) {
    append_tensor_if_valid(tensors, matmul_layer->get_bias(static_cast<int32_t>(idx)));
  }
}

bool tensor_is_within_range(const tensor::Tensor& tensor, const char* host_base, size_t host_bytes) {
  if (tensor.is_empty()) {
    return false;
  }
  auto buffer = tensor.get_buffer();
  if (!buffer || !buffer->ptr()) {
    return false;
  }
  const auto* tensor_begin = static_cast<const char*>(buffer->ptr());
  const auto* tensor_end = tensor_begin + tensor.byte_size();
  const auto* host_end = host_base + host_bytes;
  return tensor_begin >= host_base && tensor_end <= host_end;
}

std::shared_ptr<base::Buffer> make_tensor_pool_view(const tensor::Tensor& tensor, const char* host_base,
                                                    void* device_base) {
  if (tensor.is_empty()) {
    return nullptr;
  }
  auto current_buffer = tensor.get_buffer();
  if (!current_buffer || !current_buffer->ptr()) {
    return nullptr;
  }

  const auto offset = static_cast<size_t>(static_cast<const char*>(current_buffer->ptr()) - host_base);
  auto view_buffer = std::make_shared<base::Buffer>(
      tensor.byte_size(), nullptr, static_cast<char*>(device_base) + offset, true);
  view_buffer->set_device_type(base::DeviceType::kDeviceCUDA);
  return view_buffer;
}

}  // namespace

std::shared_ptr<base::Buffer> bulk_load_param_layers_to_cuda(
    const std::vector<std::shared_ptr<op::Layer>>& param_layers,
    std::shared_ptr<kernel::CudaConfig> config,
    LoadProgressCallback progress_callback,
    const void* source_base,
    size_t source_bytes) {
  if (!config || !config->stream) {
    return nullptr;
  }

  std::vector<tensor::Tensor*> tensors;
  tensors.reserve(param_layers.size() * 2);
  for (const auto& layer : param_layers) {
    collect_layer_tensors(layer, tensors);
  }
  if (tensors.empty()) {
    return nullptr;
  }

  const auto* contiguous_base = static_cast<const char*>(source_base);
  const size_t contiguous_bytes = source_bytes;
  std::vector<tensor::Tensor*> bulk_tensors;
  std::vector<tensor::Tensor*> outlier_tensors;
  bulk_tensors.reserve(tensors.size());
  outlier_tensors.reserve(tensors.size());
  for (tensor::Tensor* tensor : tensors) {
    if (tensor == nullptr) {
      return nullptr;
    }
    if (contiguous_base != nullptr && contiguous_bytes > 0 &&
        tensor_is_within_range(*tensor, contiguous_base, contiguous_bytes)) {
      bulk_tensors.push_back(tensor);
    } else {
      outlier_tensors.push_back(tensor);
    }
  }

  if (bulk_tensors.empty()) {
    return nullptr;
  }

  struct OutlierCopyGroup {
    const void* host_ptr = nullptr;
    size_t byte_size = 0;
    std::vector<tensor::Tensor*> tensors;
  };

  std::vector<OutlierCopyGroup> outlier_groups;
  outlier_groups.reserve(outlier_tensors.size());
  for (tensor::Tensor* tensor : outlier_tensors) {
    auto buffer = tensor->get_buffer();
    if (!buffer || !buffer->ptr()) {
      return nullptr;
    }
    const void* host_ptr = buffer->ptr();
    const size_t byte_size = tensor->byte_size();
    auto group_it = std::find_if(
        outlier_groups.begin(), outlier_groups.end(),
        [host_ptr, byte_size](const OutlierCopyGroup& group) {
          return group.host_ptr == host_ptr && group.byte_size == byte_size;
        });
    if (group_it == outlier_groups.end()) {
      OutlierCopyGroup group;
      group.host_ptr = host_ptr;
      group.byte_size = byte_size;
      group.tensors.push_back(tensor);
      outlier_groups.push_back(std::move(group));
    } else {
      group_it->tensors.push_back(tensor);
    }
  }

  size_t total_bytes = contiguous_bytes;
  for (const auto& group : outlier_groups) {
    total_bytes += group.byte_size;
  }

  auto cuda_allocator = base::CUDADeviceAllocatorFactory::get_instance();
  auto device_buffer = std::make_shared<base::Buffer>(contiguous_bytes, cuda_allocator);
  if (!device_buffer || !device_buffer->ptr()) {
    if (progress_callback) {
      progress_callback(0, total_bytes, "weights.bulk_alloc_failed");
    }
    LOG(WARNING) << "Skip optimized weight loading because contiguous CUDA allocation failed.";
    return nullptr;
  }

  if (progress_callback) {
    progress_callback(0, total_bytes, "weights.bulk_copy");
  }

  bool host_registered = false;
  cudaError_t register_status = cudaHostRegister(const_cast<char*>(contiguous_base), contiguous_bytes,
                                                 cudaHostRegisterPortable);
  if (register_status == cudaSuccess) {
    host_registered = true;
  } else {
    LOG(WARNING) << "cudaHostRegister failed, fallback to pageable host transfer for bulk weight "
                    "upload: "
                 << cudaGetErrorString(register_status);
    cudaGetLastError();
  }

  cudaError_t copy_status = cudaMemcpyAsync(device_buffer->ptr(), contiguous_base, contiguous_bytes,
                                            cudaMemcpyHostToDevice, config->stream);
  if (copy_status != cudaSuccess) {
    if (host_registered) {
      cudaHostUnregister(const_cast<char*>(contiguous_base));
    }
    if (progress_callback) {
      progress_callback(0, total_bytes, "weights.bulk_copy_failed");
    }
    LOG(WARNING) << "Optimized bulk weight upload failed during cudaMemcpyAsync: "
                 << cudaGetErrorString(copy_status);
    return nullptr;
  }

  cudaError_t sync_status = cudaStreamSynchronize(config->stream);
  if (host_registered) {
    cudaError_t unregister_status = cudaHostUnregister(const_cast<char*>(contiguous_base));
    if (unregister_status != cudaSuccess) {
      LOG(WARNING) << "cudaHostUnregister failed after bulk weight upload: "
                   << cudaGetErrorString(unregister_status);
    }
  }
  if (sync_status != cudaSuccess) {
    if (progress_callback) {
      progress_callback(0, total_bytes, "weights.bulk_sync_failed");
    }
    LOG(WARNING) << "Optimized bulk weight upload failed during stream synchronize: "
                 << cudaGetErrorString(sync_status);
    return nullptr;
  }

  std::vector<std::pair<tensor::Tensor*, std::shared_ptr<base::Buffer>>> rebound_views;
  rebound_views.reserve(bulk_tensors.size());
  for (tensor::Tensor* tensor : bulk_tensors) {
    auto view_buffer = make_tensor_pool_view(*tensor, contiguous_base, device_buffer->ptr());
    if (!view_buffer) {
      if (progress_callback) {
        progress_callback(0, total_bytes, "weights.bulk_rebind_prepare_failed");
      }
      LOG(WARNING) << "Optimized bulk weight upload failed while preparing a tensor view.";
      return nullptr;
    }
    rebound_views.emplace_back(tensor, std::move(view_buffer));
  }
  for (auto& entry : rebound_views) {
    if (!entry.first->assign(entry.second)) {
      if (progress_callback) {
        progress_callback(0, total_bytes, "weights.bulk_rebind_failed");
      }
      LOG(WARNING) << "Optimized bulk weight upload failed while rebinding a tensor view.";
      return nullptr;
    }
  }

  size_t loaded_bytes = contiguous_bytes;
  std::vector<std::pair<OutlierCopyGroup*, std::shared_ptr<base::Buffer>>> outlier_buffers;
  outlier_buffers.reserve(outlier_groups.size());
  std::vector<const void*> registered_outlier_ptrs;
  registered_outlier_ptrs.reserve(outlier_groups.size());
  for (auto& group : outlier_groups) {
    auto outlier_buffer = std::make_shared<base::Buffer>(group.byte_size, cuda_allocator);
    if (!outlier_buffer || !outlier_buffer->ptr()) {
      if (progress_callback) {
        progress_callback(0, total_bytes, "weights.bulk_outlier_alloc_failed");
      }
      return nullptr;
    }

    cudaError_t outlier_register =
        cudaHostRegister(const_cast<void*>(group.host_ptr), group.byte_size, cudaHostRegisterPortable);
    if (outlier_register == cudaSuccess) {
      registered_outlier_ptrs.push_back(group.host_ptr);
    } else {
      cudaGetLastError();
    }

    cudaError_t outlier_copy = cudaMemcpyAsync(outlier_buffer->ptr(), group.host_ptr, group.byte_size,
                                               cudaMemcpyHostToDevice, config->stream);
    if (outlier_copy != cudaSuccess) {
      for (const void* host_ptr : registered_outlier_ptrs) {
        cudaHostUnregister(const_cast<void*>(host_ptr));
      }
      if (progress_callback) {
        progress_callback(0, total_bytes, "weights.bulk_outlier_copy_failed");
      }
      return nullptr;
    }
    outlier_buffers.emplace_back(&group, std::move(outlier_buffer));
    loaded_bytes += group.byte_size;
    if (progress_callback) {
      progress_callback(std::min(loaded_bytes, total_bytes), total_bytes, "weights.bulk_outlier");
    }
  }
  if (!outlier_groups.empty()) {
    cudaError_t outlier_sync = cudaStreamSynchronize(config->stream);
    for (const void* host_ptr : registered_outlier_ptrs) {
      cudaHostUnregister(const_cast<void*>(host_ptr));
    }
    if (outlier_sync != cudaSuccess) {
      if (progress_callback) {
        progress_callback(0, total_bytes, "weights.bulk_outlier_sync_failed");
      }
      LOG(WARNING) << "Optimized weight loading failed while synchronizing outlier tensor copies: "
                   << cudaGetErrorString(outlier_sync);
      return nullptr;
    }
    for (auto& entry : outlier_buffers) {
      for (tensor::Tensor* tensor : entry.first->tensors) {
        if (!tensor->assign(entry.second)) {
          if (progress_callback) {
            progress_callback(0, total_bytes, "weights.bulk_outlier_rebind_failed");
          }
          return nullptr;
        }
      }
    }
  }

  LOG(INFO) << "Optimized weight loading uploaded " << (contiguous_bytes >> 20)
            << " MB of contiguous weight data with " << outlier_groups.size()
            << " outlier copy groups.";
  if (progress_callback) {
    progress_callback(total_bytes, total_bytes, "done");
  }
  return device_buffer;
}

}  // namespace model
