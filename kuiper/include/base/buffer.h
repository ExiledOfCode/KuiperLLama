// 文件说明：Buffer 抽象声明，封装跨设备内存块、持有关系和原始指针访问。

#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
// Buffer 是一段连续内存的 RAII 封装，也是 Tensor 的实际存储句柄。
//
// 所有权语义：
// - use_external=false：Buffer 通过 allocator_ 申请并在析构时释放 ptr_；
// - use_external=true：Buffer 只是外部内存视图，不释放 ptr_，常用于 mmap 权重或 KV cache slot。
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;  // 当前视图可访问的字节数。
  void* ptr_ = nullptr;   // CPU 或 CUDA 地址，device_type_ 说明地址所在设备。
  bool use_external_ = false;  // true 表示不拥有 ptr_，析构时不释放。
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_;  // 拥有内存时用于 release/copy。

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  // 延迟分配接口：构造时未传 ptr 且有 allocator 时也会自动分配。
  bool allocate();

  // 将另一个 Buffer 拷贝到当前 Buffer，拷贝方向由两个 Buffer 的 device_type 决定。
  void copy_from(const Buffer& buffer) const;

  void copy_from(const Buffer* buffer) const;

  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;

  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  std::shared_ptr<Buffer> get_shared_from_this();

  bool is_external() const;
};
}  // namespace base

#endif
