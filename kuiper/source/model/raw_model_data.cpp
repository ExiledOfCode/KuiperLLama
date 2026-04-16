#include "model/raw_model_data.h"
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
namespace model {
RawModelData::~RawModelData() {
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
  return static_cast<float*>(weight_data) + offset;
}

const void* RawModelDataInt8::weight(size_t offset) const {
  return static_cast<int8_t*>(weight_data) + offset;
}

void RawModelDataBf16::load_from_bf16(const uint16_t* source, size_t count) {
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
  source_weights = source;
  converted_weights.clear();
  weight_data = const_cast<uint16_t*>(source_weights);
}

const void* RawModelDataBf16::weight(size_t offset) const {
  if (source_weights != nullptr) {
    return source_weights + offset;
  }
  return converted_weights.data() + offset;
}
}  // namespace model
