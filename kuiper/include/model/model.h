// 文件说明：模型基类声明，抽象权重加载、前向计算、采样、KV cache 和进度回调。

#ifndef KUIPER_INCLUDE_MODEL_MODEL_H_
#define KUIPER_INCLUDE_MODEL_MODEL_H_
#include <cstdint>
#include <op/embedding.h>
#include <functional>
#include <map>
#include <string>
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/temperature_sampler.h"
#include "tensor/tensor.h"

namespace model {
class PagedKVCache;

struct OpProfileStat {
  std::string op_name;
  double total_ms = 0.0;
  int64_t calls = 0;
  double avg_ms = 0.0;
};

using LoadProgressCallback = std::function<void(size_t loaded_bytes, size_t total_bytes,
                                                const std::string& stage)>;

// 所有具体模型的抽象基类。
//
// Model 负责与模型结构无关的公共流程：
// tokenizer 创建、模型文件 mmap、权重类型识别、运行时 buffer 注册、采样器配置、
// 连续/分页 KV cache 的统一访问，以及加载进度回调。
// Qwen/Llama 等派生类只需要实现层创建、内存布局和单 token forward。
class Model {
 public:
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model);

  virtual base::Status init(base::DeviceType device_type) = 0;

  // predict 是上层生成循环调用的入口：forward 计算 logits，post_processing 采样 next token。
  virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               bool is_prompt, int& next) const = 0;

  // forward 只执行 Transformer 计算，不做 token 采样。
  virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

  int32_t max_seq_len() const;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual bool has_buffer(ModelBufferType buffer_idx) const;

  virtual bool is_sentence_ending(int32_t token_idx) const;

  virtual std::string decode(int32_t token_idx) const;

  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  // 文本侧接口：服务端 demo 会先 encode prompt，再逐 token predict，最后 decode 输出。
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  // 返回当前层、当前位置可写入的 K/V cache 视图，具体实现可为连续 cache 或分页 cache。
  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) const;

  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;

  // 从 embedding 输出中取当前位置 token 的 hidden 向量，形成单 token decode 输入。
  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

  void set_load_progress_callback(LoadProgressCallback callback);

  void set_sampling_temperature(float temperature);

  float sampling_temperature() const;

  bool optimized_weight_loading() const;

  bool paged_kv_cache_enabled() const;

  int32_t paged_kv_cache_page_size() const;

  int32_t paged_kv_cache_startup_tokens() const;

  const void* contiguous_weight_data() const;

  size_t contiguous_weight_data_byte_size() const;

 protected:
  // 模型运行时中间张量统一注册到 buffers_，派生类通过枚举键复用同一块内存。
  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  // 读取模型文件 header/config/group_size，并把权重 payload mmap 到 RawModelData。
  virtual base::Status read_model_file();

  virtual base::Status create_encode_layer();

  virtual base::Status gen_model_from_file();

  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  // 权重加载和运行时 buffer 分配会通过该回调上报到 stdout，再被 server trace/进度接口消费。
  void notify_load_progress(size_t loaded_bytes, size_t total_bytes,
                            const std::string& stage) const;

  bool use_paged_kv_cache() const;

  bool init_paged_kv_cache();

  bool ensure_paged_kv_cache(int32_t token_pos) const;

  const PagedKVCache* paged_kv_cache() const;

  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

 private:
  virtual void init_mem() = 0;

  virtual base::Status create_layers() = 0;

  virtual void create_param_layers() = 0;

  virtual void create_nonparam_layers() = 0;

  virtual void create_param_quant_layers() = 0;

 protected:
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;
  base::WeightType weight_type_ = base::WeightType::kWeightTypeFp32;
  base::DataType weight_data_type_ = base::DataType::kDataTypeFp32;
  base::DataType quant_non_param_data_type_ = base::DataType::kDataTypeFp32;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
  LoadProgressCallback load_progress_callback_;
  float sampling_temperature_ = 0.0f;
  bool optimized_weight_loading_ = false;
  bool paged_kv_cache_enabled_ = false;
  int32_t paged_kv_cache_page_size_ = 256;
  size_t weight_data_byte_size_ = 0;
  std::shared_ptr<base::Buffer> device_weight_buffer_;
  std::shared_ptr<PagedKVCache> paged_kv_cache_;
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_MODEL_H_
