// 文件说明：分页 KV Cache 声明，用按需分配的 page 降低长上下文显存占用。

#ifndef SRC_INCLUDE_MODEL_PAGED_KV_CACHE_H_
#define SRC_INCLUDE_MODEL_PAGED_KV_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "base/buffer.h"
#include "tensor/tensor.h"

namespace model {

// 分页 KV cache 以 page 为单位按需扩容。
//
// 连续 KV cache 会一次性分配 [layer_num, seq_len, kv_dim] 两块大 buffer；
// 分页模式只在访问到某个 token_pos 时确保对应 page 存在，并把 page table 提供给 CUDA MHA kernel。
// 这样长上下文模型启动时不需要立即占满最大上下文显存。
class PagedKVCache {
 public:
  PagedKVCache(base::DeviceType device_type, int32_t layer_num, int32_t max_seq_len,
               int32_t kv_dim, int32_t page_size);

  // 确保 token_pos 所在 page 已经存在，超过 max_seq_len 会返回 false。
  bool ensure_token_capacity(int32_t token_pos);

  // 返回当前 layer/token 的 K/V 写入视图。返回 Tensor 不拥有 page 内存。
  std::pair<tensor::Tensor, tensor::Tensor> slot(int32_t layer_idx, int32_t token_pos,
                                                 base::DeviceType tensor_device_type) const;

  // CUDA 模式返回设备端 page table，CPU 模式返回 host vector.data()。
  const void* key_page_table_ptr() const;

  const void* value_page_table_ptr() const;

  int32_t page_size() const;

  int32_t page_count() const;

  int32_t capacity_tokens() const;

  int32_t startup_token_capacity() const;

  size_t page_byte_size() const;

  size_t allocated_kv_byte_size() const;

 private:
  // 每次扩容会同时分配 key page 和 value page，并刷新 page table。
  bool allocate_page();

  bool refresh_page_tables();

  float* slot_ptr(const std::vector<std::shared_ptr<base::Buffer>>& pages, int32_t layer_idx,
                  int32_t token_pos) const;

 private:
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  int32_t layer_num_ = 0;
  int32_t max_seq_len_ = 0;
  int32_t kv_dim_ = 0;
  int32_t page_size_ = 1;
  size_t page_byte_size_ = 0;
  std::shared_ptr<base::DeviceAllocator> allocator_;
  std::vector<std::shared_ptr<base::Buffer>> key_pages_;
  std::vector<std::shared_ptr<base::Buffer>> value_pages_;
  std::vector<float*> key_page_ptrs_host_;
  std::vector<float*> value_page_ptrs_host_;
  std::shared_ptr<base::Buffer> key_page_table_buffer_;
  std::shared_ptr<base::Buffer> value_page_table_buffer_;
};

}  // namespace model

#endif  // SRC_INCLUDE_MODEL_PAGED_KV_CACHE_H_
