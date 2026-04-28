#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op {
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  UNUSED(idx);
  UNUSED(weight);
  return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type,
                                   base::DataType data_type) {
  UNUSED(idx);
  UNUSED(dims);
  UNUSED(weight_ptr);
  UNUSED(device_type);
  UNUSED(data_type);
  return base::error::FunctionNotImplement();
}

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }
base::DeviceType BaseLayer::device_type() const { return device_type_; }

void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

base::Status Layer::init() { return base::error::Success(); }

base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          base::DeviceType device_type, base::DataType data_type,
                                          ...) const {
  std::va_list args;
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  for (int32_t i = 0; i < dims; ++i) {
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.get_dim(i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  va_end(args);
  return base::error::Success();
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

base::Status Layer::check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }

LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer,
                       std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)),
      is_quant_layer_(is_quant_layer),
      quant_type_(is_quant_layer ? QuantType::kInt8Sym : QuantType::kNone) {}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  if (quant_type_ == QuantType::kInt8Sym || quant_type_ == QuantType::kAwqInt4) {
    CHECK(weight.data_type() == base::DataType::kDataTypeInt8);
  } else {
    CHECK(weight.data_type() == base::DataType::kDataTypeFp32 ||
          weight.data_type() == base::DataType::kDataTypeBf16);
  }
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

bool LayerParam::has_scales() const { return !scales_.is_empty(); }

tensor::Tensor& LayerParam::get_scales() { return scales_; }

const tensor::Tensor& LayerParam::get_scales() const { return scales_; }

bool LayerParam::has_zeros() const { return !zeros_.is_empty(); }

tensor::Tensor& LayerParam::get_zeros() { return zeros_; }

const tensor::Tensor& LayerParam::get_zeros() const { return zeros_; }

void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : weights_) {
    weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!zeros_.is_empty()) {
    zeros_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type,
                                    base::DataType data_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t logical_size = std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1),
                                        std::multiplies<>());
  const base::DataType target_type =
      is_quantized() ? base::DataType::kDataTypeInt8 : data_type;
  size_t size = logical_size * base::DataTypeSize(target_type);
  if (quant_type_ == QuantType::kAwqInt4) {
    size = (logical_size + 1) / 2;
  }
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quantized()) {
    tensor::Tensor weight(target_type, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
  } else if (quant_type_ == QuantType::kInt8Sym) {
    tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(logical_size);
    CHECK(weight_size % group_size_ == 0);

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.set_device_type(device_type);
  } else if (quant_type_ == QuantType::kAwqInt4) {
    CHECK_EQ(logical_size % static_cast<size_t>(group_size_), 0)
        << "AWQ int4 weights require numel to be divisible by group_size.";
    const size_t packed_size = (logical_size + 1) / 2;
    tensor::Tensor weight(base::DataType::kDataTypeInt8, static_cast<int32_t>(packed_size));
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t scale_nums = static_cast<int32_t>(logical_size / group_size_);
    const auto* scales_ptr = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>(weight_ptr) + packed_size);
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             const_cast<float*>(scales_ptr)};
    scales_.set_device_type(device_type);

    const auto* zeros_ptr =
        reinterpret_cast<const int8_t*>(scales_ptr + scale_nums);
    zeros_ = tensor::Tensor{base::DataType::kDataTypeInt8, scale_nums, false, nullptr,
                            const_cast<int8_t*>(zeros_ptr)};
    zeros_.set_device_type(device_type);
  }

  return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

void LayerParam::set_quant_type(QuantType quant_type) {
  quant_type_ = quant_type;
  is_quant_layer_ = quant_type != QuantType::kNone;
}

QuantType LayerParam::quant_type() const { return quant_type_; }

bool LayerParam::is_quantized() const { return quant_type_ != QuantType::kNone; }

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }

size_t LayerParam::weight_byte_size() const {
  size_t total = 0;
  for (const auto& weight : weights_) {
    if (!weight.is_empty()) {
      total += weight.byte_size();
    }
  }
  if (!scales_.is_empty()) {
    total += scales_.byte_size();
  }
  if (!zeros_.is_empty()) {
    total += zeros_.byte_size();
  }
  return total;
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

}  // namespace op
