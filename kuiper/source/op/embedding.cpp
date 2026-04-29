// 文件说明：Embedding 算子实现，根据 token id 从词嵌入表取出隐藏向量。

#include "op/embedding.h"
#include "kernels/cpu/emb_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"
namespace op {
namespace {

base::Status check_embedding_weight(const tensor::Tensor& tensor, base::DeviceType device_type,
                                    int32_t vocab_size, int32_t dim) {
  // embedding 表支持 FP32/BF16；输出统一为 data_type_ 指定的运行时类型。
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != base::DataType::kDataTypeFp32 &&
      tensor.data_type() != base::DataType::kDataTypeBf16) {
    return base::error::InvalidArgument("The tensor has an unsupported weight data type.");
  }
  if (tensor.dims_size() != 2 || tensor.get_dim(0) != vocab_size || tensor.get_dim(1) != dim) {
    return base::error::InvalidArgument("The tensor has a wrong dim.");
  }
  return base::error::Success();
}

}  // namespace

EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
  reset_weight_size(1);
  reset_input_size(2);
  reset_output_size(1);
}

base::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  // input(1) 只用 size 表示有效 token 数；input_tensor 的容量可能更大。
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
  }

  base::Status status = check_tensor_with_dim(input_tensor, base::DeviceType::kDeviceCPU,
                                              base::DataType::kDataTypeInt32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  status = check_embedding_weight(get_weight(0), device_type_, vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_size, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }
  return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  // CPU/CUDA kernel 都按 token id 逐个 gather embedding 行。
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::StatusCode::kSuccess;
}
}  // namespace op
