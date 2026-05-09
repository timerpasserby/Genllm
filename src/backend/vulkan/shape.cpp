#include "backend/vulkan/shape.h"
#include <stdexcept>

namespace ops {

void ReshapeImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan ReshapeImpl: not implemented");
}
void PermuteImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan PermuteImpl: not implemented");
}
void ConcatImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan ConcatImpl: not implemented");
}
void RepeatImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan RepeatImpl: not implemented");
}

template struct ReshapeImpl<Device::VULKAN>;
template struct PermuteImpl<Device::VULKAN>;
template struct ConcatImpl<Device::VULKAN>;
template struct RepeatImpl<Device::VULKAN>;

}
