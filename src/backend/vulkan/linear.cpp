#include "backend/vulkan/linear.h"
#include <stdexcept>

namespace ops {

void MatmulImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan MatmulImpl: not implemented");
}
void LinearImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan LinearImpl: not implemented");
}
void TransposeImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan TransposeImpl: not implemented");
}

template struct MatmulImpl<Device::VULKAN>;
template struct LinearImpl<Device::VULKAN>;
template struct TransposeImpl<Device::VULKAN>;

}
