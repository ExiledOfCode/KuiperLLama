// 文件说明：多头注意力 kernel 实现，计算 QK softmax 和 V 加权求和。

#ifndef LLAMA_INFER_MHA_KERNEL_H
#define LLAMA_INFER_MHA_KERNEL_H
#include <base/cuda_config.h>
#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                const void* key_page_table, const void* value_page_table, int32_t page_size,
                bool paged_kv_cache, base::DeviceType device_type, CudaConfig* config);
}  // namespace kernel
#endif  // LLAMA_INFER_MHA_KERNEL_H
