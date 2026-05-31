// 文件说明：Qwen3 模型适配声明，覆盖 Qwen3 的归一化、注意力、FFN 和量化权重路径。

#ifndef SRC_INCLUDE_MODEL_LLAMA_H_
#define SRC_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {
// Qwen3 导出配置的模型侧补充结构，保留给 Qwen3 专用解析或调试使用。
// 当前运行主路径使用通用 TransformerConfig。
struct QWen3TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t immediate_size_ = 0;
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
};

// Qwen3 的所有算子层句柄集合。
// 参数层按 block 顺序分别存放，非参数层（RoPE/MHA/Add/SwiGLU）在所有 block 间复用。
struct Qwen3Layers {
  // 无权重算子：只依赖输入张量和配置。
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  // Attention 投影矩阵，每层一组 Wq/Wk/Wv/Wo。
  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  // FFN 矩阵：Qwen3 使用 SwiGLU，因此有 gate/up/down 三个投影。
  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  // rmsnorm_layers_ 的布局：
  // [0, layer_num)               attention RMSNorm
  // [layer_num, 2*layer_num)     FFN RMSNorm
  // [2*layer_num]                final output RMSNorm
  // [2*layer_num+1, 3*layer_num+1) query RMSNorm
  // [3*layer_num+1, 4*layer_num+1) key RMSNorm
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  // 将层内参数迁移到 CUDA。优化路径会返回一块由所有权重视图共享的显存池。
  std::shared_ptr<base::Buffer> to_cuda(std::shared_ptr<kernel::CudaConfig> config,
                                        LoadProgressCallback progress_callback,
                                        bool optimized_weight_loading,
                                        const void* contiguous_weight_data,
                                        size_t contiguous_weight_bytes);
};

class Qwen3Model : public Model {
 public:
  explicit Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  // 将 token id 批量转成 embedding，prompt 阶段一次处理整个 prompt。
  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  void set_profile_enabled(bool enabled) const;

  void reset_profile_stats() const;

  std::vector<OpProfileStat> get_profile_stats() const;

 private:
  // 分配所有推理过程中会复用的中间 Tensor，包括 KV cache、logits 和临时激活。
  void init_mem() override;

  base::Status create_layers() override;

  // 按权重文件导出顺序创建并绑定参数层。量化模型走 create_param_quant_layers。
  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  // Qwen3 block 内的三个主要阶段。
  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<Qwen3Layers> qwen_layers_;
};
}  // namespace model

#endif
