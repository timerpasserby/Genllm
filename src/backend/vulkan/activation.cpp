#include "backend/vulkan/activation.h"
#include <stdexcept>

namespace ops {

void SiluImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan SiluImpl: not implemented");
}
void GeluImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan GeluImpl: not implemented");
}
void ReluImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan ReluImpl: not implemented");
}

template struct SiluImpl<Device::VULKAN>;
template struct GeluImpl<Device::VULKAN>;
template struct ReluImpl<Device::VULKAN>;

}
