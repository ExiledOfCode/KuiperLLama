// 文件说明：Embedding kernel 实现，按 token 索引拷贝词嵌入向量。

//
// Created by fss on 24-6-7.
//

#ifndef SRC_INFER_EMB_KERNEL_H
#define SRC_INFER_EMB_KERNEL_H
#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
void emb_kernel_normal(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size,
                       void* stream = nullptr);
}  // namespace kernel
#endif  // SRC_INFER_EMB_KERNEL_H
