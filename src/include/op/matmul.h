// 文件说明：矩阵乘层声明，覆盖全精度、INT8、BF16 和 AWQ INT4 权重乘法。

//
// Created by hello on 2024/5/2.
//

#ifndef SRC_INCLUDE_OP_MATMUL_H_
#define SRC_INCLUDE_OP_MATMUL_H_
#include <base/cuda_config.h>
#include "layer.h"
namespace op {
// MatmulLayer 表示 y = W * x (+ bias)。
// dim0 是输出维度，dim1 是输入维度；权重形状为 [dim0, dim1]。
// 量化路径复用同一接口，通过 QuantType 选择 INT8 对称量化或 AWQ INT4 kernel。
class MatmulLayer : public LayerParam {
 public:
  explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                       bool is_quant_layer = false, bool has_bias = false);

  static std::shared_ptr<MatmulLayer> create_awq_int4(base::DeviceType device_type, int32_t dim0,
                                                      int32_t dim1,
                                                      bool has_bias = false);

  base::Status check() const override;

  base::Status forward() override;

  base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                        base::DeviceType device_type,
                        base::DataType data_type = base::DataType::kDataTypeFp32);

  tensor::Tensor& get_bias(int32_t idx);

  const tensor::Tensor& get_bias(int32_t idx) const;

  bool has_bias() const;

  size_t bias_size() const;

  void to_cuda() override;

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
  bool has_bias_ = false;
  std::vector<tensor::Tensor> bias_;
};
}  // namespace op
#endif  // SRC_INCLUDE_OP_MATMUL_H_
