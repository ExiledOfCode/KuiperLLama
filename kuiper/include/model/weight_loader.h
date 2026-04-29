// 文件说明：权重批量加载接口声明，优化连续权重向 CUDA 显存上传的路径。

#ifndef KUIPER_INCLUDE_MODEL_WEIGHT_LOADER_H_
#define KUIPER_INCLUDE_MODEL_WEIGHT_LOADER_H_

#include <memory>
#include <vector>

#include "model/model.h"

namespace model {

// 将一组参数层的权重上传到 CUDA：
// 1. 如果权重 tensor 都落在同一段 mmap payload 中，先整段拷到一块显存；
// 2. 再按原始 offset 给每个 tensor 绑定显存视图；
// 3. 少数不在连续 payload 内的 outlier tensor 单独拷贝并 rebind。
//
// 返回值是连续显存池，调用方必须持有它，避免 tensor 视图悬空。返回 nullptr 表示优化路径失败，
// 调用方应回退到逐层 to_cuda()。
std::shared_ptr<base::Buffer> bulk_load_param_layers_to_cuda(
    const std::vector<std::shared_ptr<op::Layer>>& param_layers,
    std::shared_ptr<kernel::CudaConfig> config,
    LoadProgressCallback progress_callback,
    const void* source_base,
    size_t source_bytes);

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_WEIGHT_LOADER_H_
