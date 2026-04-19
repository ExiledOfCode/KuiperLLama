#include "model/paged_kv_cache.h"

#include <algorithm>

#include <glog/logging.h>

#include "base/alloc.h"

namespace model {

namespace {

std::shared_ptr<base::DeviceAllocator> get_allocator(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return base::CUDADeviceAllocatorFactory::get_instance();
  }
  return base::CPUDeviceAllocatorFactory::get_instance();
}

std::shared_ptr<base::Buffer> make_slot_view(float* slot_ptr, size_t byte_size,
                                             base::DeviceType device_type) {
  if (slot_ptr == nullptr || byte_size == 0) {
    return nullptr;
  }
  auto buffer = std::make_shared<base::Buffer>(byte_size, nullptr, slot_ptr, true);
  buffer->set_device_type(device_type);
  return buffer;
}

}  // namespace

PagedKVCache::PagedKVCache(base::DeviceType device_type, int32_t layer_num, int32_t max_seq_len,
                           int32_t kv_dim, int32_t page_size)
    : device_type_(device_type),
      layer_num_(std::max(0, layer_num)),
      max_seq_len_(std::max(1, max_seq_len)),
      kv_dim_(std::max(1, kv_dim)),
      page_size_(std::max(1, std::min(page_size, std::max(1, max_seq_len)))),
      allocator_(get_allocator(device_type)) {
  page_byte_size_ =
      static_cast<size_t>(layer_num_) * static_cast<size_t>(page_size_) *
      static_cast<size_t>(kv_dim_) * sizeof(float);
}

bool PagedKVCache::ensure_token_capacity(int32_t token_pos) {
  if (token_pos < 0 || token_pos >= max_seq_len_) {
    LOG(ERROR) << "PagedKVCache token position out of range: " << token_pos
               << ", max_seq_len=" << max_seq_len_;
    return false;
  }

  const int32_t required_pages = token_pos / page_size_ + 1;
  while (page_count() < required_pages) {
    if (!allocate_page()) {
      return false;
    }
  }
  return true;
}

std::pair<tensor::Tensor, tensor::Tensor> PagedKVCache::slot(
    int32_t layer_idx, int32_t token_pos, base::DeviceType tensor_device_type) const {
  CHECK_GE(layer_idx, 0);
  CHECK_LT(layer_idx, layer_num_);
  CHECK_GE(token_pos, 0);
  CHECK_LT(token_pos, max_seq_len_);

  const size_t slot_byte_size = static_cast<size_t>(kv_dim_) * sizeof(float);
  float* key_ptr = slot_ptr(key_pages_, layer_idx, token_pos);
  float* value_ptr = slot_ptr(value_pages_, layer_idx, token_pos);
  CHECK_NE(key_ptr, nullptr);
  CHECK_NE(value_ptr, nullptr);

  tensor::Tensor key(base::DataType::kDataTypeFp32, kv_dim_);
  tensor::Tensor value(base::DataType::kDataTypeFp32, kv_dim_);
  CHECK(key.assign(make_slot_view(key_ptr, slot_byte_size, tensor_device_type)));
  CHECK(value.assign(make_slot_view(value_ptr, slot_byte_size, tensor_device_type)));
  key.set_device_type(tensor_device_type);
  value.set_device_type(tensor_device_type);
  return {key, value};
}

const void* PagedKVCache::key_page_table_ptr() const {
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    return key_page_table_buffer_ ? key_page_table_buffer_->ptr() : nullptr;
  }
  return key_page_ptrs_host_.empty() ? nullptr : key_page_ptrs_host_.data();
}

const void* PagedKVCache::value_page_table_ptr() const {
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    return value_page_table_buffer_ ? value_page_table_buffer_->ptr() : nullptr;
  }
  return value_page_ptrs_host_.empty() ? nullptr : value_page_ptrs_host_.data();
}

int32_t PagedKVCache::page_size() const { return page_size_; }

int32_t PagedKVCache::page_count() const { return static_cast<int32_t>(key_pages_.size()); }

int32_t PagedKVCache::capacity_tokens() const { return page_count() * page_size_; }

int32_t PagedKVCache::startup_token_capacity() const { return page_size_; }

size_t PagedKVCache::page_byte_size() const { return page_byte_size_; }

size_t PagedKVCache::allocated_kv_byte_size() const {
  return static_cast<size_t>(page_count()) * page_byte_size_ * 2;
}

bool PagedKVCache::allocate_page() {
  auto key_page = std::make_shared<base::Buffer>(page_byte_size_, allocator_);
  auto value_page = std::make_shared<base::Buffer>(page_byte_size_, allocator_);
  if (!key_page || !value_page || key_page->ptr() == nullptr || value_page->ptr() == nullptr) {
    LOG(ERROR) << "Failed to allocate a paged KV cache page.";
    return false;
  }
  key_pages_.push_back(key_page);
  value_pages_.push_back(value_page);
  key_page_ptrs_host_.push_back(static_cast<float*>(key_page->ptr()));
  value_page_ptrs_host_.push_back(static_cast<float*>(value_page->ptr()));
  return refresh_page_tables();
}

bool PagedKVCache::refresh_page_tables() {
  if (device_type_ != base::DeviceType::kDeviceCUDA) {
    return true;
  }

  const size_t table_bytes = static_cast<size_t>(page_count()) * sizeof(float*);
  if (table_bytes == 0) {
    return true;
  }

  if (!key_page_table_buffer_ || key_page_table_buffer_->byte_size() < table_bytes) {
    key_page_table_buffer_ = std::make_shared<base::Buffer>(table_bytes, allocator_);
  }
  if (!value_page_table_buffer_ || value_page_table_buffer_->byte_size() < table_bytes) {
    value_page_table_buffer_ = std::make_shared<base::Buffer>(table_bytes, allocator_);
  }
  if (!key_page_table_buffer_ || !value_page_table_buffer_ ||
      key_page_table_buffer_->ptr() == nullptr || value_page_table_buffer_->ptr() == nullptr) {
    LOG(ERROR) << "Failed to allocate paged KV cache page tables.";
    return false;
  }

  allocator_->memcpy(key_page_ptrs_host_.data(), key_page_table_buffer_->ptr(), table_bytes,
                     base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  allocator_->memcpy(value_page_ptrs_host_.data(), value_page_table_buffer_->ptr(), table_bytes,
                     base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  return true;
}

float* PagedKVCache::slot_ptr(const std::vector<std::shared_ptr<base::Buffer>>& pages,
                              int32_t layer_idx, int32_t token_pos) const {
  const int32_t page_idx = token_pos / page_size_;
  const int32_t page_token = token_pos % page_size_;
  CHECK_GE(page_idx, 0);
  CHECK_LT(page_idx, static_cast<int32_t>(pages.size()));
  CHECK(pages.at(page_idx) != nullptr);
  CHECK(pages.at(page_idx)->ptr() != nullptr);

  float* page_base = static_cast<float*>(pages.at(page_idx)->ptr());
  const size_t layer_offset =
      static_cast<size_t>(layer_idx) * static_cast<size_t>(page_size_) *
      static_cast<size_t>(kv_dim_);
  const size_t token_offset =
      static_cast<size_t>(page_token) * static_cast<size_t>(kv_dim_);
  return page_base + layer_offset + token_offset;
}

}  // namespace model
