#include "backend/vulkan/normalization.h"
#include <stdexcept>

namespace ops {

void RmsNormImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan RmsNormImpl: not implemented");
}
void LayerNormImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan LayerNormImpl: not implemented");
}

template struct RmsNormImpl<Device::VULKAN>;
template struct LayerNormImpl<Device::VULKAN>;

}
