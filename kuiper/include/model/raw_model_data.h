// 文件说明：原始权重数据视图声明，封装 mmap 权重、量化权重和 BF16 转换访问。

#ifndef RAW_MODEL_DATA_H
#define RAW_MODEL_DATA_H
#include <cstddef>
#include <cstdint>
#include <vector>
namespace model {
// RawModelData 只描述权重文件/权重区间的生命周期和寻址方式：
// - data 指向完整 mmap 文件；
// - weight_data 指向去掉 header/config/group_size 后的权重 payload 起始地址；
// - weight(offset) 由具体权重格式解释 offset 的单位。
struct RawModelData {
  ~RawModelData();
  int32_t fd = -1;         // mmap 对应的文件描述符，由析构函数关闭。
  size_t file_size = 0;    // mmap 的完整文件大小。
  void* data = nullptr;    // mmap 起始地址。
  void* weight_data = nullptr;  // 权重 payload 起始地址，类型由派生类解释。

  virtual const void* weight(size_t offset) const = 0;
};

// FP32 权重 offset 以 float 元素为单位。
struct RawModelDataFp32 : RawModelData {
  const void* weight(size_t offset) const override;
};

// INT8 对称量化权重 offset 以 byte/int8 元素为单位。
struct RawModelDataInt8 : RawModelData {
  const void* weight(size_t offset) const override;
};

// AWQ INT4 权重以两个 4-bit 值打包到一个 byte，offset 仍按打包后的 byte 地址前进。
struct RawModelDataAwqInt4 : RawModelData {
  const void* weight(size_t offset) const override;
};

// BF16 有两条路径：
// - 支持原生 BF16 的 CUDA 设备直接指向 mmap 中的 uint16_t；
// - CPU 或不支持 BF16 的设备启动时扩展为 converted_weights 中的 FP32。
struct RawModelDataBf16 : RawModelData {
  void load_from_bf16(const uint16_t* source, size_t count);

  void use_source_weights(const uint16_t* source);

  const void* weight(size_t offset) const override;

  const uint16_t* source_weights = nullptr;
  std::vector<float> converted_weights;
};

}  // namespace model
#endif  // RAW_MODEL_DATA_H
