// 文件说明：词嵌入层声明，将 token id 映射为模型隐藏向量。


#ifndef KUIPER_INCLUDE_OP_EMBEDDING_H_
#define KUIPER_INCLUDE_OP_EMBEDDING_H_
#include <utility>
#include "layer.h"
namespace op {
// EmbeddingOutput 把 embedding 层相关的三个 Tensor 一起返回：
// token ids、embedding 结果和 token 数量。后续 fill_input 会从 input_embeddings 切出单 token。
struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
  explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num)
      : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

// EmbeddingLayer 根据 input token id 从 [vocab_size, dim] 权重表中查出隐藏向量。
class EmbeddingLayer : public LayerParam {
 public:
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_EMBEDDING_H_
