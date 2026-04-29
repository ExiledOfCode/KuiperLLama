// 文件说明：原始权重数据实现，提供 mmap 释放、量化偏移访问和 BF16 权重转换。

#include "model/raw_model_data.h"
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
namespace model {
RawModelData::~RawModelData() {
  // RawModelData 拥有完整模型文件的 mmap 生命周期；各层 Tensor 只是指向 weight_data 的外部视图。
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, file_size);
    data = nullptr;
  }
  if (fd != -1) {
    close(fd);
    fd = -1;
  }
}

const void* RawModelDataFp32::weight(size_t offset) const {
  // FP32/BF16 转 FP32 后 offset 都以 float 元素计数。
  return static_cast<float*>(weight_data) + offset;
}

const void* RawModelDataInt8::weight(size_t offset) const {
  // INT8 量化权重按 byte 地址偏移。
  return static_cast<int8_t*>(weight_data) + offset;
}

const void* RawModelDataAwqInt4::weight(size_t offset) const {
  // AWQ INT4 已经打包为 byte 流，调用方负责按 packed_size 推进 offset。
  return static_cast<uint8_t*>(weight_data) + offset;
}

void RawModelDataBf16::load_from_bf16(const uint16_t* source, size_t count) {
  // 不支持原生 BF16 的设备会在加载阶段一次性扩展到 FP32，后续层仍按 float 权重访问。
  source_weights = nullptr;
  converted_weights.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const uint32_t bits = static_cast<uint32_t>(source[i]) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(float));
    converted_weights[i] = value;
  }
  weight_data = converted_weights.data();
}

void RawModelDataBf16::use_source_weights(const uint16_t* source) {
  // sm80+ CUDA kernel 可直接消费 BF16，此时保留 mmap 中的原始 uint16_t 权重，避免双份内存。
  source_weights = source;
  converted_weights.clear();
  weight_data = const_cast<uint16_t*>(source_weights);
}

const void* RawModelDataBf16::weight(size_t offset) const {
  if (source_weights != nullptr) {
    // 原生 BF16 路径 offset 以 uint16_t 元素计数。
    return source_weights + offset;
  }
  return converted_weights.data() + offset;
}
}  // namespace model
