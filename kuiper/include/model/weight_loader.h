#ifndef KUIPER_INCLUDE_MODEL_WEIGHT_LOADER_H_
#define KUIPER_INCLUDE_MODEL_WEIGHT_LOADER_H_

#include <memory>
#include <vector>

#include "model/model.h"

namespace model {

std::shared_ptr<base::Buffer> bulk_load_param_layers_to_cuda(
    const std::vector<std::shared_ptr<op::Layer>>& param_layers,
    std::shared_ptr<kernel::CudaConfig> config,
    LoadProgressCallback progress_callback,
    const void* source_base,
    size_t source_bytes);

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_WEIGHT_LOADER_H_
