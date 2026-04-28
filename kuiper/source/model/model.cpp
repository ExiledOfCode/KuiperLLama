#include "model/model.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "model/paged_kv_cache.h"
namespace model {
namespace {

bool env_flag_enabled(const char* env_name, bool default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr) {
    return default_value;
  }
  std::string text(raw);
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (text == "1" || text == "true" || text == "yes" || text == "on") {
    return true;
  }
  if (text == "0" || text == "false" || text == "no" || text == "off") {
    return false;
  }
  return default_value;
}

int32_t env_positive_int(const char* env_name, int32_t default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr) {
    return default_value;
  }
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || value <= 0) {
    return default_value;
  }
  if (value > std::numeric_limits<int32_t>::max()) {
    return default_value;
  }
  return static_cast<int32_t>(value);
}

bool cuda_device_supports_bf16() {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    LOG(WARNING) << "Failed to query current CUDA device: " << cudaGetErrorString(err);
    return false;
  }

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    LOG(WARNING) << "Failed to query CUDA device properties: " << cudaGetErrorString(err);
    return false;
  }
  return prop.major >= 8;
}

base::DataType quant_non_param_dtype_from_header_reserved(int32_t reserved) {
  const auto data_type = static_cast<base::DataType>(reserved);
  if (data_type == base::DataType::kDataTypeBf16 ||
      data_type == base::DataType::kDataTypeFp32) {
    return data_type;
  }
  return base::DataType::kDataTypeFp32;
}

}  // namespace

Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model),
      optimized_weight_loading_(env_flag_enabled("KLLM_OPTIMIZED_WEIGHT_LOADING", false)),
      paged_kv_cache_enabled_(env_flag_enabled("KLLM_PAGED_KV_CACHE", false)),
      paged_kv_cache_page_size_(env_positive_int("KLLM_PAGED_KV_CACHE_PAGE_SIZE", 256)) {
  LOG(INFO) << "Optimized weight loading is "
            << (optimized_weight_loading_ ? "enabled" : "disabled")
            << " for model: " << model_path_;
  LOG(INFO) << "Paged KV cache is " << (paged_kv_cache_enabled_ ? "enabled" : "disabled")
            << " for model: " << model_path_ << ", page_size=" << paged_kv_cache_page_size_;
}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

int32_t Model::max_seq_len() const { return config_ ? config_->seq_len_ : 0; }

void Model::set_load_progress_callback(LoadProgressCallback callback) {
  load_progress_callback_ = std::move(callback);
}

void Model::set_sampling_temperature(float temperature) {
  if (temperature < 0.0f) {
    sampling_temperature_ = 0.0f;
    return;
  }
  sampling_temperature_ = temperature;
}

float Model::sampling_temperature() const { return sampling_temperature_; }

bool Model::optimized_weight_loading() const { return optimized_weight_loading_; }

bool Model::paged_kv_cache_enabled() const { return paged_kv_cache_enabled_; }

int32_t Model::paged_kv_cache_page_size() const { return paged_kv_cache_page_size_; }

int32_t Model::paged_kv_cache_startup_tokens() const {
  if (config_ == nullptr) {
    return std::max(1, paged_kv_cache_page_size_);
  }
  return std::max(1, std::min(config_->seq_len_, paged_kv_cache_page_size_));
}

const void* Model::contiguous_weight_data() const {
  return raw_model_data_ ? raw_model_data_->weight_data : nullptr;
}

size_t Model::contiguous_weight_data_byte_size() const { return weight_data_byte_size_; }

void Model::notify_load_progress(size_t loaded_bytes, size_t total_bytes,
                                 const std::string& stage) const {
  if (load_progress_callback_) {
    load_progress_callback_(loaded_bytes, total_bytes, stage);
  }
}

bool Model::use_paged_kv_cache() const { return paged_kv_cache_enabled_; }

bool Model::init_paged_kv_cache() {
  if (!use_paged_kv_cache() || config_ == nullptr) {
    return false;
  }
  if (!paged_kv_cache_) {
    paged_kv_cache_ = std::make_shared<PagedKVCache>(
        device_type_, config_->layer_num_, config_->seq_len_, config_->kv_dim_,
        paged_kv_cache_startup_tokens());
  }
  return paged_kv_cache_->ensure_token_capacity(0);
}

bool Model::ensure_paged_kv_cache(int32_t token_pos) const {
  if (!use_paged_kv_cache()) {
    return true;
  }
  if (!paged_kv_cache_) {
    LOG(ERROR) << "Paged KV cache is enabled but has not been initialized.";
    return false;
  }
  return paged_kv_cache_->ensure_token_capacity(token_pos);
}

const PagedKVCache* Model::paged_kv_cache() const { return paged_kv_cache_.get(); }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

bool Model::has_buffer(ModelBufferType buffer_idx) const { return buffers_.count(buffer_idx) > 0; }

base::Status Model::read_model_file() {
  using namespace base;
  weight_data_byte_size_ = 0;
  device_weight_buffer_.reset();
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    close(fd);
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  ModelFileHeader file_header{};
  auto config = ModelConfig{};
  size_t weight_data_offset = 0;

  if (fread(&file_header, sizeof(ModelFileHeader), 1, file) == 1 &&
      file_header.magic == kModelFileMagic) {
    if (file_header.version != kModelFileVersion) {
      fclose(file);
      close(fd);
      return error::ModelParseError("Unsupported model file header version.");
    }
    weight_type_ = static_cast<base::WeightType>(file_header.weight_type);
    is_quant_model_ = weight_type_ == base::WeightType::kWeightTypeInt8 ||
                      weight_type_ == base::WeightType::kWeightTypeAwqInt4;
    quant_non_param_data_type_ =
        quant_non_param_dtype_from_header_reserved(file_header.reserved);
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
      fclose(file);
      close(fd);
      return error::ModelParseError(
          "Failed to retrieve the configuration information from the model "
          "file.");
    }
    weight_data_offset = sizeof(ModelFileHeader) + sizeof(ModelConfig);
  } else {
    std::rewind(file);
    weight_type_ = is_quant_model_ ? base::WeightType::kWeightTypeInt8
                                   : base::WeightType::kWeightTypeFp32;
    quant_non_param_data_type_ = base::DataType::kDataTypeFp32;
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
      fclose(file);
      close(fd);
      return error::ModelParseError(
          "Failed to retrieve the configuration information from the model "
          "file.");
    }
    weight_data_offset = sizeof(ModelConfig);
  }

  if (weight_type_ == base::WeightType::kWeightTypeInt8 ||
      weight_type_ == base::WeightType::kWeightTypeAwqInt4) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      fclose(file);
      close(fd);
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
    weight_data_offset += sizeof(group_size_);
  }
  fclose(file);

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    close(fd);
    return gen_status;
  }

  if (weight_type_ == base::WeightType::kWeightTypeFp32) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
    weight_data_type_ = base::DataType::kDataTypeFp32;
  } else if (weight_type_ == base::WeightType::kWeightTypeInt8) {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
    weight_data_type_ = base::DataType::kDataTypeInt8;
  } else if (weight_type_ == base::WeightType::kWeightTypeAwqInt4) {
    raw_model_data_ = std::make_shared<RawModelDataAwqInt4>();
    weight_data_type_ = base::DataType::kDataTypeInt8;
  } else if (weight_type_ == base::WeightType::kWeightTypeBf16) {
    raw_model_data_ = std::make_shared<RawModelDataBf16>();
    weight_data_type_ = base::DataType::kDataTypeBf16;
  } else {
    close(fd);
    return error::ModelParseError("Unsupported weight type in model file.");
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  raw_model_data_->file_size = sb.st_size;

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }
  if (weight_type_ == base::WeightType::kWeightTypeFp32 ||
      weight_type_ == base::WeightType::kWeightTypeInt8 ||
      weight_type_ == base::WeightType::kWeightTypeAwqInt4) {
    raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + weight_data_offset;
    weight_data_byte_size_ = raw_model_data_->file_size - weight_data_offset;
  } else {
    const size_t payload_bytes = raw_model_data_->file_size - weight_data_offset;
    if (payload_bytes % sizeof(uint16_t) != 0) {
      return error::ModelParseError("The bf16 model payload is malformed.");
    }
    auto bf16_data = std::static_pointer_cast<RawModelDataBf16>(raw_model_data_);
    const auto* source = reinterpret_cast<const uint16_t*>(
        static_cast<int8_t*>(raw_model_data_->data) + weight_data_offset);
    if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_device_supports_bf16()) {
      bf16_data->use_source_weights(source);
      weight_data_type_ = base::DataType::kDataTypeBf16;
      weight_data_byte_size_ = payload_bytes;
      LOG(INFO) << "Enable native CUDA BF16 weight path for model: " << model_path_;
    } else {
      bf16_data->load_from_bf16(source, payload_bytes / sizeof(uint16_t));
      weight_data_type_ = base::DataType::kDataTypeFp32;
      weight_data_byte_size_ = bf16_data->converted_weights.size() * sizeof(float);
      if (device_type_ == base::DeviceType::kDeviceCUDA) {
        LOG(WARNING) << "Current CUDA device does not support native BF16. "
                     << "Fallback to fp32 weights for model: " << model_path_;
      }
    }
  }
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
  config_->immediate_dim_ = config.immediate_dim_;
#endif
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
#if !defined(QWEN2_SUPPORT) && !defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
#else
    return error::InternalError(
        "SentencePiece tokenizer is disabled in Qwen-only build. "
        "Please use Qwen tokenizer JSON.");
#endif
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  if (use_paged_kv_cache()) {
    CHECK_NE(paged_kv_cache_, nullptr);
    return paged_kv_cache_->slot(layer_idx, token_pos, device_type_);
  }
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
#if defined(QWEN3_SUPPORT)
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      config_->hidden_dim_ * sizeof(float), nullptr,
      input_embeddings.ptr<float>(index * config_->hidden_dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->hidden_dim_);

#else
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * config_->dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
#endif
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model
