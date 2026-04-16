#include "op/rmsnorm.h"
#include <cuda_runtime_api.h>
#include <armadillo>
#include "kernels/cpu/rmsnorm_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
namespace {

base::Status check_rmsnorm_weight(const tensor::Tensor& tensor, base::DeviceType device_type,
                                  int32_t dim) {
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
  if (tensor.dims_size() != 1 || tensor.get_dim(0) != dim) {
    return base::error::InvalidArgument("The tensor has a wrong dim.");
  }
  return base::error::Success();
}

}  // namespace

RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::kLayerRMSNorm, false, "RMSNorm"), dim_(dim) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status RmsNormLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  if (input.dims_size() == 1) {
    kernel::get_rmsnorm_kernel(device_type_)(input, weight, output,
                                             cuda_config_ ? cuda_config_->stream : nullptr);
  } else {
    kernel::get_rmsnorm_dim_kernel(device_type_)(input, weight, output, dim_,
                                                 cuda_config_ ? cuda_config_->stream : nullptr);
  }

  return base::error::Success();
}

base::Status RmsNormLayer::check() const {
  int32_t dim_size = get_input(0).dims_size();
  if (dim_size > 1) {
    int dim_head_size = get_input(0).get_dim(dim_size - 1);
    if (dim_head_size == dim_) {
      return base::error::Success();
    } else {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim -1");
    }
  } else {
    auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
    if (!status) {
      LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
      return status;
    }

    status = check_rmsnorm_weight(get_weight(0), device_type_, dim_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the rmsnorm layer.";
      return status;
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
    if (!status) {
      LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
      return status;
    }
    return base::error::Success();
  }
}

}  // namespace op
