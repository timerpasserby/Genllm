#include "backend/vulkan/rope.h"
#include <stdexcept>

namespace ops {

void ApplyRopeImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan ApplyRopeImpl: not implemented");
}

template struct ApplyRopeImpl<Device::VULKAN>;

}
