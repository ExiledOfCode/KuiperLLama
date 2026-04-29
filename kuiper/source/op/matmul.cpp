// 文件说明：矩阵乘算子实现，根据权重类型和设备分发到对应 kernel。

#include "op/matmul.h"
#include <cstdint>
#include <cstring>
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
namespace {

base::Status check_weight_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 int32_t dim0, int32_t dim1) {
  // 非量化 matmul 接受 FP32/BF16 权重，形状固定为 [out_dim, in_dim]。
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
  if (tensor.dims_size() != 2 || tensor.get_dim(0) != dim0 || tensor.get_dim(1) != dim1) {
    return base::error::InvalidArgument("The tensor has a wrong dim.");
  }
  return base::error::Success();
}

float bf16_to_fp32(uint16_t value) {
  // BF16 的高 16 位对应 FP32 的符号/指数/高位尾数，低 16 位补零即可得到近似 FP32。
  uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

}  // namespace

MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
      dim0_(dim0),
      dim1_(dim1),
      has_bias_(has_bias) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
  if (has_bias_) {
    bias_.resize(1);
  }
}

std::shared_ptr<MatmulLayer> MatmulLayer::create_awq_int4(base::DeviceType device_type,
                                                          int32_t dim0, int32_t dim1,
                                                          bool has_bias) {
  // AWQ INT4 仍复用 MatmulLayer，只切换 quant_type 以影响权重解析、check 和 kernel 分发。
  auto layer = std::make_shared<MatmulLayer>(device_type, dim0, dim1, false, has_bias);
  layer->set_quant_type(QuantType::kAwqInt4);
  return layer;
}

base::Status MatmulLayer::check() const {
  // 输入是一维向量 [dim1]，输出是一维向量 [dim0]；batch/prompt 由外层逐 token 驱动。
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  if (quant_type_ == QuantType::kNone) {
    status = check_weight_tensor(get_weight(0), device_type_, dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  } else if (quant_type_ == QuantType::kInt8Sym) {
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                   dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  } else if (quant_type_ == QuantType::kAwqInt4) {
    // packed weight 只有一个维度，长度为 ceil(dim0 * dim1 / 2)。
    const int32_t packed_size = (dim0_ * dim1_ + 1) / 2;
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                   packed_size);
    if (!status) {
      LOG(ERROR) << "The AWQ int4 packed weight tensor error in the matmul layer.";
      return status;
    }
  }

  if (is_quantized()) {
    status = check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32,
                                   static_cast<int32_t>(scales_.size()));
    if (!status) {
      LOG(ERROR) << "The scale tensor error in the matmul layer.";
      return status;
    }
    if (quant_type_ == QuantType::kAwqInt4) {
      status = check_tensor_with_dim(zeros_, device_type_, base::DataType::kDataTypeInt8,
                                     static_cast<int32_t>(zeros_.size()));
      if (!status) {
        LOG(ERROR) << "The zero tensor error in the matmul layer.";
        return status;
      }
    }
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  // 根据权重格式选择 kernel；调用者不需要关心 FP32/BF16/INT8/AWQ 的分发细节。
  if (quant_type_ == QuantType::kInt8Sym) {
    kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                   group_size_, scales_,
                                                   cuda_config_ ? cuda_config_.get() : nullptr);
  } else if (quant_type_ == QuantType::kAwqInt4) {
    kernel::get_matmul_kernel_awq_int4(device_type_)(
        get_input(0), get_weight(0), get_output(0), group_size_, scales_, zeros_,
        cuda_config_ ? cuda_config_.get() : nullptr);
  } else {
    kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                            cuda_config_ ? cuda_config_.get() : nullptr);
  }

  if (has_bias_) {
    // bias 作为额外向量加到 matmul 输出上，直接复用 add kernel。
    kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0),
                                            cuda_config_ ? cuda_config_->stream : nullptr);
  }

  return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                   base::DeviceType device_type, base::DataType data_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  if (!is_quant_layer_) {
    CHECK(data_type == base::DataType::kDataTypeFp32 || data_type == base::DataType::kDataTypeBf16);
    if (data_type == base::DataType::kDataTypeBf16) {
      // bias 目前按 FP32 参与计算，因此 BF16 bias 在加载时转换为 FP32。
      auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
      tensor::Tensor bias(base::DataType::kDataTypeFp32, dim, true, cpu_alloc);
      bias.set_device_type(base::DeviceType::kDeviceCPU);
      const auto* src = reinterpret_cast<const uint16_t*>(bias_ptr);
      float* dst = bias.ptr<float>();
      for (int32_t i = 0; i < dim; ++i) {
        dst[i] = bf16_to_fp32(src[i]);
      }
      bias_.at(idx) = bias;
    } else {
      // FP32 bias 直接绑定外部 Buffer，和权重一样不在加载阶段复制。
      size_t size = dim * sizeof(float);
      std::shared_ptr<base::Buffer> buffer =
          std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
      if (device_type != base::DeviceType::kDeviceUnknown) {
        buffer->set_device_type(device_type);
      }
      tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
      bias.set_device_type(device_type);
      CHECK(bias.assign(buffer));
      // LOG(INFO) << "bias:" << bias.index<float>(0);
      bias_.at(idx) = bias;
    }
  } else {
    // is quant layer
    // 量化 bias 复用 INT8 + scale 的布局，scale 紧跟 bias payload。
    size_t size = dim * sizeof(float);
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
    if (device_type != base::DeviceType::kDeviceUnknown) {
      buffer->set_device_type(device_type);
    }
    tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;

    const int32_t bias_size = static_cast<int32_t>(bias.size());
    CHECK(bias_size % group_size_ == 0);

    int32_t scale_nums = bias_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

bool MatmulLayer::has_bias() const { return has_bias_ && !bias_.empty(); }

size_t MatmulLayer::bias_size() const { return bias_.size(); }

void MatmulLayer::to_cuda() {
  LayerParam::to_cuda();
  if (has_bias_) {
    for (auto& bias : bias_) {
      bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

}  // namespace op
