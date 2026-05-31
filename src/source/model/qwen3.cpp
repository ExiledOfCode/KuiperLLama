// 文件说明：Qwen3 模型实现，支持 BF16/INT8/AWQ 权重、推理跟踪和分页 KV Cache。
#define QWEN3_SUPPORT
#ifdef QWEN3_SUPPORT
#include "model/qwen3.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "base/tick.h"
#include "model/paged_kv_cache.h"
#include "model/weight_loader.h"

namespace {

struct ProfileAccumulator {
  double total_ms = 0.0;
  int64_t calls = 0;
};

// 每个线程独立记录 profile，避免服务端多实例/测试并行时互相污染。
thread_local std::unordered_map<std::string, ProfileAccumulator> g_qwen3_op_profile;
thread_local bool g_qwen3_profile_enabled = true;

double elapsed_ms(const std::chrono::steady_clock::time_point& begin,
                  const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

size_t dims_byte_size(base::DataType data_type, std::initializer_list<size_t> dims) {
  size_t total = base::DataTypeSize(data_type);
  for (const size_t dim : dims) {
    total *= dim;
  }
  return total;
}

size_t qwen3_rope_fill_progress_bytes(const model::TransformerConfig& config) {
  return 2 * dims_byte_size(base::DataType::kDataTypeFp32,
                            {static_cast<size_t>(config.head_size_),
                             static_cast<size_t>(config.seq_len_)});
}

size_t qwen3_runtime_progress_bytes(const model::TransformerConfig& config,
                                    int32_t kv_cache_tokens) {
  // 估算运行时 buffer 分配量，用于服务端加载进度。该值只用于展示，不参与内存分配。
  size_t total = 0;
  total += dims_byte_size(base::DataType::kDataTypeInt32, {1});
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {1, static_cast<size_t>(config.hidden_dim_)});
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {static_cast<size_t>(config.head_size_),
                           static_cast<size_t>(config.seq_len_)}) *
           2;
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {static_cast<size_t>(config.hidden_dim_)});
  total += dims_byte_size(base::DataType::kDataTypeFp32, {static_cast<size_t>(config.dim_)});
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {static_cast<size_t>(config.immediate_dim_)}) *
           2;
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {static_cast<size_t>(config.layer_num_),
                           static_cast<size_t>(kv_cache_tokens),
                           static_cast<size_t>(config.kv_dim_)}) *
           2;
  total += dims_byte_size(base::DataType::kDataTypeFp32, {static_cast<size_t>(config.dim_)});
  total += dims_byte_size(base::DataType::kDataTypeInt32, {1});
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {static_cast<size_t>(config.head_num_),
                           static_cast<size_t>(config.seq_len_)});
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {static_cast<size_t>(config.hidden_dim_)});
  total += dims_byte_size(base::DataType::kDataTypeFp32,
                          {static_cast<size_t>(config.vocab_size_)});
  total += qwen3_rope_fill_progress_bytes(config);
  return total;
}

void record_profile(const char* op_name, double cost_ms) {
  if (!g_qwen3_profile_enabled || op_name == nullptr || *op_name == '\0') {
    return;
  }
  auto& item = g_qwen3_op_profile[op_name];
  item.total_ms += cost_ms;
  item.calls += 1;
}

template <class Fn>
base::Status run_profiled(const char* op_name, Fn&& fn) {
  // 用模板包装 lambda，避免每个算子调用点手写计时和 profile 汇总代码。
  const auto begin = std::chrono::steady_clock::now();
  base::Status status = fn();
  const auto end = std::chrono::steady_clock::now();
  record_profile(op_name, elapsed_ms(begin, end));
  return status;
}

std::vector<model::OpProfileStat> snapshot_profile_stats() {
  // 返回时按总耗时降序，前端/日志能直接看到最耗时的算子。
  std::vector<model::OpProfileStat> stats;
  stats.reserve(g_qwen3_op_profile.size());
  for (const auto& [name, value] : g_qwen3_op_profile) {
    model::OpProfileStat item;
    item.op_name = name;
    item.total_ms = value.total_ms;
    item.calls = value.calls;
    item.avg_ms = value.calls > 0 ? (value.total_ms / static_cast<double>(value.calls)) : 0.0;
    stats.push_back(item);
  }
  std::sort(stats.begin(), stats.end(),
            [](const model::OpProfileStat& lhs, const model::OpProfileStat& rhs) {
              if (lhs.total_ms == rhs.total_ms) {
                return lhs.op_name < rhs.op_name;
              }
              return lhs.total_ms > rhs.total_ms;
            });
  return stats;
}

}  // namespace

namespace model {

std::shared_ptr<base::Buffer> Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config,
                                                   LoadProgressCallback progress_callback,
                                                   bool optimized_weight_loading,
                                                   const void* contiguous_weight_data,
                                                   size_t contiguous_weight_bytes) {
  // 参数层收集顺序不改变模型语义，只用于统一设置 CUDA stream、统计字节数和尝试批量上传。
  auto layer_weight_bytes = [](const std::shared_ptr<op::Layer>& layer) -> size_t {
    auto param_layer = std::dynamic_pointer_cast<op::LayerParam>(layer);
    return param_layer ? param_layer->weight_byte_size() : 0;
  };

  std::vector<std::shared_ptr<op::Layer>> param_layers;
  auto collect_layer = [&param_layers](const std::shared_ptr<op::Layer>& layer) {
    if (layer) {
      param_layers.push_back(layer);
    }
  };
  auto collect_layers = [&collect_layer](const std::vector<std::shared_ptr<op::Layer>>& layers) {
    for (const auto& layer : layers) {
      collect_layer(layer);
    }
  };

  collect_layer(cls_layer_);
  collect_layer(embedding_layer_);
  collect_layers(wq_layers_);
  collect_layers(wk_layers_);
  collect_layers(wv_layers_);
  collect_layers(wo_layers_);
  collect_layers(w1_layers_);
  collect_layers(w2_layers_);
  collect_layers(w3_layers_);
  collect_layers(rmsnorm_layers_);

  for (const auto& layer : param_layers) {
    if (layer) {
      layer->set_cuda_config(config);
    }
  }

  size_t total_bytes = 0;
  for (const auto& layer : param_layers) {
    total_bytes += layer_weight_bytes(layer);
  }
  size_t loaded_bytes = 0;

  auto copy_layer = [&](const std::shared_ptr<op::Layer>& layer, const std::string& stage) {
    // legacy 路径逐层调用 to_cuda，每完成一层按该层权重大小推进进度。
    if (!layer) {
      return;
    }
    const size_t bytes = layer_weight_bytes(layer);
    layer->to_cuda();
    if (bytes > 0) {
      loaded_bytes += bytes;
      if (progress_callback) {
        progress_callback(std::min(loaded_bytes, total_bytes), total_bytes, stage);
      }
    }
  };

  if (progress_callback) {
    progress_callback(0, total_bytes, "start");
  }

  // 非参数层只需要绑定 CUDA 上下文；真正占大头的权重层在后面统一上传。
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  if (optimized_weight_loading) {
    if (progress_callback) {
      progress_callback(0, contiguous_weight_bytes, "weights.bulk_prepare");
    }
    // mmap 权重通常是连续内存，批量上传后再把各层 tensor 重新绑定到同一块显存视图。
    auto device_buffer = bulk_load_param_layers_to_cuda(param_layers, config, progress_callback,
                                                        contiguous_weight_data, contiguous_weight_bytes);
    if (device_buffer) {
      return device_buffer;
    }
    if (progress_callback) {
      progress_callback(0, total_bytes, "weights.bulk_fallback");
    }
    LOG(WARNING) << "Falling back to legacy per-layer CUDA weight upload for Qwen3.";
  }

  if (cls_layer_) {
    copy_layer(cls_layer_, "lm_head");
  }

  if (embedding_layer_) {
    copy_layer(embedding_layer_, "embedding");
  }

  for (auto& weight_layer : wq_layers_) {
    copy_layer(weight_layer, "attention.wq");
  }

  for (auto& weight_layer : wk_layers_) {
    copy_layer(weight_layer, "attention.wk");
  }

  for (auto& weight_layer : wv_layers_) {
    copy_layer(weight_layer, "attention.wv");
  }

  for (auto& weight_layer : wo_layers_) {
    copy_layer(weight_layer, "attention.wo");
  }

  for (auto& weight_layer : w1_layers_) {
    copy_layer(weight_layer, "ffn.w1");
  }

  for (auto& weight_layer : w2_layers_) {
    copy_layer(weight_layer, "ffn.w2");
  }

  for (auto& weight_layer : w3_layers_) {
    copy_layer(weight_layer, "ffn.w3");
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    copy_layer(rms_norm_layer, "rmsnorm");
  }

  if (progress_callback) {
    progress_callback(total_bytes, total_bytes, "done");
  }
  return nullptr;
}

Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status Qwen3Model::init(base::DeviceType device_type) {
  using namespace base;
  // init 是模型生命周期入口：检查路径/设备、创建 CUDA stream、读取权重、分配运行时 buffer。
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  // 层已经绑定到 mmap 权重，此处再按设备创建中间 buffer，并按需把参数迁移到 CUDA。
  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    // RoPE sin/cos cache 是纯配置相关数据，初始化后整个请求生命周期复用。
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    const size_t weight_total_bytes = contiguous_weight_data_byte_size();
    const int32_t kv_cache_tokens =
        use_paged_kv_cache() ? paged_kv_cache_startup_tokens() : config_->seq_len_;
    const size_t runtime_total_bytes = qwen3_runtime_progress_bytes(*config_, kv_cache_tokens);
    const size_t rope_fill_bytes = qwen3_rope_fill_progress_bytes(*config_);
    const size_t overall_total_bytes = weight_total_bytes + runtime_total_bytes;
    notify_load_progress(overall_total_bytes - rope_fill_bytes, overall_total_bytes,
                         "buffers.rope_cache_fill");
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
    notify_load_progress(overall_total_bytes, overall_total_bytes, "done");
  }

  sampler_ = std::make_unique<sampler::TemperatureSampler>(device_type_, sampling_temperature_);
  return error::Success();
}

base::Status Qwen3Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  // 单 token decode 路径：每层依次执行 attn rmsnorm -> qkv/rope/cache -> attention -> FFN。
  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    attention_mha(layer_idx, pos_tensor);
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

void Qwen3Model::create_nonparam_layers() {
  CHECK(qwen_layers_ != nullptr);
  // 非参数层不持有权重，创建一次后在所有 Transformer block 间复用。
  qwen_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  qwen_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->immediate_dim_);
}

void Qwen3Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(qwen_layers_ != nullptr);

  // pos 按导出脚本写入顺序在 payload 中前进。量化层的实际步长由 weight_byte_size() 给出，
  // 因为 INT8/AWQ 后面还紧跟 scales/zeros。
  size_t pos = 0;
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t hidden_dim = config_->hidden_dim_;
  const int32_t immediate_dim = config_->immediate_dim_;
  const auto cpu_device_type = base::DeviceType::kDeviceCPU;
  const auto non_param_data_type = quant_non_param_data_type_;

  auto non_param_bytes = [non_param_data_type](std::initializer_list<int32_t> dims) {
    // 量化模型中的 RMSNorm/Embedding 等非矩阵参数可能是 FP32 或 BF16。
    size_t bytes = base::DataTypeSize(non_param_data_type);
    for (int32_t dim : dims) {
      bytes *= static_cast<size_t>(dim);
    }
    return bytes;
  };
  auto create_quant_matmul = [&](int32_t out_dim, int32_t in_dim) {
    // MatmulLayer 负责解释 INT8 或 AWQ INT4 的具体 payload 布局。
    std::shared_ptr<op::MatmulLayer> layer;
    if (weight_type_ == base::WeightType::kWeightTypeAwqInt4) {
      layer = op::MatmulLayer::create_awq_int4(device_type_, out_dim, in_dim);
    } else {
      layer = std::make_shared<op::MatmulLayer>(device_type_, out_dim, in_dim, true);
    }
    layer->set_group_size(group_size_);
    layer->set_weight(0, {out_dim, in_dim}, this->raw_model_data_->weight(pos),
                      cpu_device_type);
    pos += layer->weight_byte_size();
    return layer;
  };

  // rmsnorm attention, ffn, final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, hidden_dim);

    rms_norm_layer->set_weight(0, {hidden_dim}, raw_model_data_->weight(pos), cpu_device_type,
                               non_param_data_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += non_param_bytes({hidden_dim});
  }

  // embedding layer
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, hidden_dim, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), hidden_dim},
                                             raw_model_data_->weight(pos), cpu_device_type,
                                             non_param_data_type);
  pos += non_param_bytes({std::abs(config_->vocab_size_), hidden_dim});

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = create_quant_matmul(dim, hidden_dim);
    qwen_layers_->wq_layers_.push_back(wq);
  }

  // query norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type, non_param_data_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += non_param_bytes({config_->head_size_});
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = create_quant_matmul(kv_dim, hidden_dim);
    qwen_layers_->wk_layers_.push_back(wk);
  }

  // key norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type, non_param_data_type);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += non_param_bytes({config_->head_size_});
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = create_quant_matmul(kv_dim, hidden_dim);
    qwen_layers_->wv_layers_.push_back(wv);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = create_quant_matmul(hidden_dim, dim);
    qwen_layers_->wo_layers_.push_back(wo);
  }

  // w1 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = create_quant_matmul(immediate_dim, hidden_dim);
    qwen_layers_->w1_layers_.push_back(w1);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = create_quant_matmul(hidden_dim, immediate_dim);
    qwen_layers_->w2_layers_.push_back(w2);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = create_quant_matmul(immediate_dim, hidden_dim);
    qwen_layers_->w3_layers_.push_back(w3);
  }

  auto lm_head = create_quant_matmul(config_->vocab_size_, hidden_dim);
  qwen_layers_->cls_layer_ = lm_head;
}

void Qwen3Model::create_param_layers() {
  CHECK(qwen_layers_ != nullptr);

  // 全精度/BF16 权重 offset 以元素为单位推进；RawModelData 根据权重类型解释 offset。
  size_t pos = 0;
  int32_t dim = config_->dim_;
  int32_t kv_dim = config_->kv_dim_;
  int hidden_dim = config_->hidden_dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // rmsnorm attention attention, ffn, final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, hidden_dim);

    rms_norm_layer->set_weight(0, {hidden_dim}, raw_model_data_->weight(pos), cpu_device_type,
                               weight_data_type_);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos += hidden_dim;
  }

  // embedding layer
  qwen_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, hidden_dim, config_->seq_len_, std::abs(config_->vocab_size_));
  qwen_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), hidden_dim},
                                             raw_model_data_->weight(pos), cpu_device_type,
                                             weight_data_type_);
  pos += config_->vocab_size_ * hidden_dim;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, false);
    wq->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   weight_data_type_);
    qwen_layers_->wq_layers_.push_back(wq);
    pos = pos + hidden_dim * dim;
  }

  // query norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type, weight_data_type_);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, hidden_dim, false);
    wk->set_weight(0, {kv_dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   weight_data_type_);
    qwen_layers_->wk_layers_.push_back(wk);
    pos = pos + hidden_dim * kv_dim;
  }

  // key norm
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->head_size_);
    rms_norm_layer->set_weight(0, {config_->head_size_}, this->raw_model_data_->weight(pos),
                               cpu_device_type, weight_data_type_);
    qwen_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    pos = pos + config_->head_size_;
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, hidden_dim, false);
    wv->set_weight(0, {kv_dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   weight_data_type_);
    qwen_layers_->wv_layers_.push_back(wv);
    pos += kv_dim * hidden_dim;
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, false);
    wo->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type,
                   weight_data_type_);
    qwen_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * hidden_dim;
  }

  // w1 layers
  int32_t immediate_dim = config_->immediate_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
    w1->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type, weight_data_type_);
    qwen_layers_->w1_layers_.push_back(w1);
    pos = pos + hidden_dim * immediate_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, immediate_dim, false);
    w2->set_weight(0, {hidden_dim, immediate_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type, weight_data_type_);
    qwen_layers_->w2_layers_.push_back(w2);
    pos = pos + immediate_dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
    w3->set_weight(0, {immediate_dim, hidden_dim}, this->raw_model_data_->weight(pos),
                   cpu_device_type, weight_data_type_);
    qwen_layers_->w3_layers_.push_back(w3);
    pos = pos + immediate_dim * hidden_dim;
  }

  auto lm_head = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_,
                                                   config_->hidden_dim_, false);
  lm_head->set_weight(0, {config_->vocab_size_, config_->hidden_dim_},
                      this->raw_model_data_->weight(pos), cpu_device_type, weight_data_type_);
  qwen_layers_->cls_layer_ = lm_head;
}

void Qwen3Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  size_t weight_total_bytes = 0;
  size_t runtime_total_bytes = 0;
  size_t runtime_loaded_bytes = 0;
  const bool paged_kv_cache = use_paged_kv_cache();
  const int32_t kv_cache_tokens =
      paged_kv_cache ? paged_kv_cache_startup_tokens() : config_->seq_len_;
  auto emit_runtime_progress = [&](size_t delta_bytes, const std::string& stage) {
    // 运行时 buffer 进度叠加在权重进度之后，形成一个单调递增的总体加载进度。
    if (runtime_total_bytes == 0) {
      return;
    }
    runtime_loaded_bytes = std::min(runtime_loaded_bytes + delta_bytes, runtime_total_bytes);
    notify_load_progress(weight_total_bytes + runtime_loaded_bytes,
                         weight_total_bytes + runtime_total_bytes, stage);
  };

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    weight_total_bytes = contiguous_weight_data_byte_size();
    runtime_total_bytes = qwen3_runtime_progress_bytes(*config_, kv_cache_tokens);
    // 权重加载进度和运行时 buffer 分配进度合并成同一个总体进度，供服务端轮询展示。
    device_weight_buffer_ = qwen_layers_->to_cuda(
        cuda_config_,
        [this, runtime_total_bytes](size_t loaded_bytes, size_t total_bytes, const std::string& stage) {
          const size_t weight_total_bytes = total_bytes;
          const size_t overall_total_bytes = weight_total_bytes + runtime_total_bytes;
          const std::string mapped_stage = stage == "done" ? "weights.done" : stage;
          notify_load_progress(std::min(loaded_bytes, weight_total_bytes), overall_total_bytes,
                               mapped_stage);
        }, optimized_weight_loading(), contiguous_weight_data(), contiguous_weight_data_byte_size());
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  // input_embeddings 在 prompt 阶段可能 reshape 成 [prompt_len, hidden_dim]，decode 阶段回到单 token。
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->hidden_dim_, true,
                                  alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  emit_runtime_progress(
      dims_byte_size(base::DataType::kDataTypeFp32,
                     {static_cast<size_t>(config_->head_size_),
                      static_cast<size_t>(config_->seq_len_)}) *
          2,
      "buffers.rope_cache_alloc");

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));
  emit_runtime_progress(
      dims_byte_size(base::DataType::kDataTypeInt32, {1}) +
          dims_byte_size(base::DataType::kDataTypeFp32,
                         {1, static_cast<size_t>(config_->hidden_dim_)}),
      "buffers.io");

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor out_mha(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, out_mha));
  // 这些中间结果生命周期不重叠，复用同一个 rms_output Buffer 降低显存占用。
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->immediate_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->immediate_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));
  emit_runtime_progress(
      dims_byte_size(base::DataType::kDataTypeFp32,
                     {static_cast<size_t>(config_->hidden_dim_)}) +
          dims_byte_size(base::DataType::kDataTypeFp32,
                         {static_cast<size_t>(config_->dim_)}) +
          dims_byte_size(base::DataType::kDataTypeFp32,
                         {static_cast<size_t>(config_->immediate_dim_)}) *
              2,
      "buffers.activations");

  // KV cache 可按完整 seq_len 预分配，也可按 page 延迟扩容以降低长上下文启动显存。
  const size_t kv_stage_bytes = dims_byte_size(base::DataType::kDataTypeFp32,
                                               {static_cast<size_t>(config_->layer_num_),
                                                static_cast<size_t>(kv_cache_tokens),
                                                static_cast<size_t>(config_->kv_dim_)});
  if (paged_kv_cache) {
    CHECK(init_paged_kv_cache());
    emit_runtime_progress(kv_stage_bytes, "buffers.key_cache");
    emit_runtime_progress(kv_stage_bytes, "buffers.value_cache");
  } else {
    tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_,
                             config_->seq_len_, config_->kv_dim_, true, alloc);
    tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_,
                               config_->seq_len_, config_->kv_dim_, true, alloc);

    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    emit_runtime_progress(kv_stage_bytes, "buffers.key_cache");
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));
    emit_runtime_progress(kv_stage_bytes, "buffers.value_cache");
  }

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  tensor::Tensor attn_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));
  emit_runtime_progress(
      dims_byte_size(base::DataType::kDataTypeFp32, {static_cast<size_t>(config_->dim_)}) +
          dims_byte_size(base::DataType::kDataTypeInt32, {1}) +
          dims_byte_size(base::DataType::kDataTypeFp32,
                         {static_cast<size_t>(config_->head_num_),
                          static_cast<size_t>(config_->seq_len_)}) +
          dims_byte_size(base::DataType::kDataTypeFp32,
                         {static_cast<size_t>(config_->hidden_dim_)}),
      "buffers.attention");

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    // 采样器/调试路径可能需要 CPU logits，因此额外保留一块主机输出缓存。
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
  emit_runtime_progress(
      dims_byte_size(base::DataType::kDataTypeFp32,
                     {static_cast<size_t>(config_->vocab_size_)}),
      "buffers.logits");
}

base::Status Qwen3Model::create_layers() {
  using namespace base;
  if (!qwen_layers_) {
    qwen_layers_ = std::make_unique<Qwen3Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  // 创建完所有层后立即做结构校验，尽早暴露导出顺序、层数或量化格式错误。
  if (!qwen_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (qwen_layers_->rmsnorm_layers_.size() != 4 * config_->layer_num_ + 1) {
    // input norm
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
      qwen_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->wq_layers_.at(i) || !qwen_layers_->wk_layers_.at(i) ||
        !qwen_layers_->wv_layers_.at(i) || !qwen_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
      qwen_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!qwen_layers_->w1_layers_.at(i) || !qwen_layers_->w2_layers_.at(i) ||
        !qwen_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!qwen_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!qwen_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!qwen_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!qwen_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

void Qwen3Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // attn rmsnorm
  // 每层 block 的第一步：对残差流做 RMSNorm，结果供 Wq/Wk/Wv 共用。
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(
      run_profiled("attn.rmsnorm", [&]() { return rmsnorm_layer->forward(input, rmsnorm_output); }));
}

void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // key/value 直接写入当前 token 的 cache slot；后续 MHA 会读取历史所有 slot。
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(run_profiled("attn.wq", [&]() { return query_layer->forward(rmsnorm_output, query); }));

  // query norm
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  // Qwen3 对每个 head 的 query/key 单独做 RMSNorm，因此先 reshape 成 [head, head_size]。
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  STATUS_CHECK(run_profiled("attn.q_norm", [&]() { return query_norm->forward(query, query); }));
  query.reshape({(int32_t)query.size()});

  // key
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(run_profiled("attn.wk", [&]() { return key_layer->forward(rmsnorm_output, key); }));

  // key norm
  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
  STATUS_CHECK(run_profiled("attn.k_norm", [&]() { return key_norm->forward(key, key); }));
  key.reshape({(int32_t)key.size()});

  // value
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(run_profiled("attn.wv", [&]() { return value_layer->forward(rmsnorm_output, val); }));

  // rope
  CHECK_NE(qwen_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(run_profiled("attn.rope", [&]() {
    return qwen_layers_->rope_layer_->forward(query, key, pos_tensor,
                                              get_buffer(ModelBufferType::kSinCache),
                                              get_buffer(ModelBufferType::kCosCache), tensor::Tensor{});
  }));
}

base::Status Qwen3Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                 bool is_prompt, int& next) const {
  if (use_paged_kv_cache()) {
    // 分页 cache 必须在写入当前 token K/V 之前保证 page 存在。
    CHECK(ensure_paged_kv_cache(pos_tensor.index<int32_t>(0)));
  }
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void Qwen3Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // mha
  // MHA 读取当前位置之前已经写入的 query/key/value，并把结果写到 kOutputMHA。
  tensor::Tensor key_cache;
  tensor::Tensor val_cache;

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = qwen_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  auto mha = std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer);
  CHECK_NE(mha, nullptr);
  if (use_paged_kv_cache()) {
    CHECK_NE(paged_kv_cache(), nullptr);
    // 分页模式下 kernel 通过 page table 找历史 K/V；连续模式则直接传入整块 cache tensor。
    mha->set_paged_kv_cache(paged_kv_cache()->key_page_table_ptr(),
                            paged_kv_cache()->value_page_table_ptr(),
                            paged_kv_cache()->page_size(), true);
  } else {
    key_cache = get_buffer(ModelBufferType::kKeyCache);
    val_cache = get_buffer(ModelBufferType::kValueCache);
    mha->set_paged_kv_cache(nullptr, nullptr, 0, false);
  }
  int pos = pos_tensor.index<int32_t>(0);
  mha->set_pos(pos);
  mha->set_layer_idx(layer_idx);
  STATUS_CHECK(run_profiled(
      "attn.mha", [&]() { return mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output); }));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(run_profiled("attn.wo", [&]() { return wo_layer->forward(mha_output, attn_output); }));
}

void Qwen3Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // residual add
  // 第一处残差：attention 输出加回 block 输入，结果直接写回 input。
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(run_profiled("ffn.residual_add1", [&]() {
    return qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input);
  }));

  // ffn rmsnorm (post attention layernorm)
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(
      run_profiled("ffn.rmsnorm", [&]() { return ffn_rmsnorm->forward(input, ffn_norm_output); }));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(run_profiled("ffn.w1", [&]() { return w1_layer->forward(ffn_norm_output, w1_output); }));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(run_profiled("ffn.w3", [&]() { return w3_layer->forward(ffn_norm_output, w3_ouput); }));

  // SwiGLU
  // SwiGLU 使用 w1 作为 gate、w3 作为 up projection，激活后原地写回 w1_output。
  CHECK_NE(qwen_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(run_profiled("ffn.swiglu", [&]() {
    return qwen_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output);
  }));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = qwen_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(run_profiled("ffn.w2", [&]() { return w2_layer->forward(w1_output, w2_output); }));

  // residual add
  CHECK_NE(qwen_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(run_profiled("ffn.residual_add2", [&]() {
    return qwen_layers_->add_layer_->forward(input, w2_output, input);
  }));
}

op::EmbeddingOutput Qwen3Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    // prompt 长度变化时复用原 Buffer；容量不够才由 Tensor::reshape 触发重新分配。
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->hidden_dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !qwen_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(run_profiled("input.embedding", [&]() {
    return qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings);
  }));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void Qwen3Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  // 最后一层 RMSNorm 后接 lm_head，输出 vocab_size 维 logits。
  const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(run_profiled("output.rmsnorm", [&]() { return norm->forward(input, input); }));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(qwen_layers_->cls_layer_, nullptr);
  STATUS_CHECK(
      run_profiled("output.cls", [&]() { return qwen_layers_->cls_layer_->forward(input, forward_output); }));
}

int32_t Qwen3Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    // prompt prefill 阶段只填 KV cache，不从 logits 采样；外层会继续喂下一个 prompt token。
    next = -1;
  } else {
    const auto begin = std::chrono::steady_clock::now();
    next = static_cast<int32_t>(
        sampler_->sample(forward_logits, forward_output.size(), cuda_config_ ? cuda_config_->stream : nullptr));
    const auto end = std::chrono::steady_clock::now();
    record_profile("sampler.argmax", elapsed_ms(begin, end));
  }
  return next;
}

void Qwen3Model::set_profile_enabled(bool enabled) const {
  g_qwen3_profile_enabled = enabled;
  if (!enabled) {
    g_qwen3_op_profile.clear();
  }
}

void Qwen3Model::reset_profile_stats() const { g_qwen3_op_profile.clear(); }

std::vector<OpProfileStat> Qwen3Model::get_profile_stats() const {
  if (!g_qwen3_profile_enabled) {
    return {};
  }
  return snapshot_profile_stats();
}

}  // namespace model

#endif
