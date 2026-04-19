#include "op/mha.h"
#include "kernels/cpu/mha_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                       int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                       int32_t head_num, int32_t head_size)
    : Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status MultiHeadAttention::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  const tensor::Tensor& mha_out = this->get_output(0);
  const tensor::Tensor& query_tensor = this->get_input(0);
  const tensor::Tensor& score_tensor = this->get_input(1);
  const tensor::Tensor& key_cache_tensor = this->get_input(2);
  const tensor::Tensor& value_cache_tensor = this->get_input(3);

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_mha_kernel(device_type_)(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_,
                                       head_size_, mha_out, query_tensor, score_tensor,
                                       key_cache_tensor, value_cache_tensor, key_page_table_,
                                       value_page_table_, page_size_, paged_kv_cache_enabled_,
                                       device_type_, cuda_config_ ? cuda_config_.get() : nullptr);
  return base::error::Success();
}

void MultiHeadAttention::set_pos(int32_t pos) { this->pos_ = pos; }

void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->layer_index_ = layer_idx; }

void MultiHeadAttention::set_paged_kv_cache(const void* key_page_table, const void* value_page_table,
                                            int32_t page_size, bool enabled) {
  key_page_table_ = key_page_table;
  value_page_table_ = value_page_table;
  page_size_ = page_size;
  paged_kv_cache_enabled_ = enabled;
}

base::Status MultiHeadAttention::check() const {
  base::Status status;
  const int32_t checked_input_num = paged_kv_cache_enabled_ ? 2 : 4;
  for (int32_t i = 0; i < checked_input_num; ++i) {
    // mha score tensor
    status = check_tensor(get_input(i), device_type_, data_type_);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the matmul layer.";
      return status;
    }
  }
  if (paged_kv_cache_enabled_) {
    if (key_page_table_ == nullptr || value_page_table_ == nullptr || page_size_ <= 0) {
      return base::error::InvalidArgument("The paged kv cache metadata is invalid.");
    }
  }
  return check_tensor(get_output(0), device_type_, data_type_);
}

}  // namespace op
