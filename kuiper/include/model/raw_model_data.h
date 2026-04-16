#ifndef RAW_MODEL_DATA_H
#define RAW_MODEL_DATA_H
#include <cstddef>
#include <cstdint>
#include <vector>
namespace model {
struct RawModelData {
  ~RawModelData();
  int32_t fd = -1;
  size_t file_size = 0;
  void* data = nullptr;
  void* weight_data = nullptr;

  virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;
};

struct RawModelDataBf16 : RawModelData {
  void load_from_bf16(const uint16_t* source, size_t count);

  void use_source_weights(const uint16_t* source);

  const void* weight(size_t offset) const override;

  const uint16_t* source_weights = nullptr;
  std::vector<float> converted_weights;
};

}  // namespace model
#endif  // RAW_MODEL_DATA_H
