#ifndef KUIPER_INCLUDE_MODEL_PAGED_KV_CACHE_H_
#define KUIPER_INCLUDE_MODEL_PAGED_KV_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "base/buffer.h"
#include "tensor/tensor.h"

namespace model {

class PagedKVCache {
 public:
  PagedKVCache(base::DeviceType device_type, int32_t layer_num, int32_t max_seq_len,
               int32_t kv_dim, int32_t page_size);

  bool ensure_token_capacity(int32_t token_pos);

  std::pair<tensor::Tensor, tensor::Tensor> slot(int32_t layer_idx, int32_t token_pos,
                                                 base::DeviceType tensor_device_type) const;

  const void* key_page_table_ptr() const;

  const void* value_page_table_ptr() const;

  int32_t page_size() const;

  int32_t page_count() const;

  int32_t capacity_tokens() const;

  int32_t startup_token_capacity() const;

  size_t page_byte_size() const;

  size_t allocated_kv_byte_size() const;

 private:
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

#endif  // KUIPER_INCLUDE_MODEL_PAGED_KV_CACHE_H_
