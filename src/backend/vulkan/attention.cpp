#include "backend/vulkan/attention.h"
#include <stdexcept>

namespace ops {

void SoftmaxImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan SoftmaxImpl: not implemented");
}
void DiagMaskInfImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan DiagMaskInfImpl: not implemented");
}
void SdpaImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    throw std::runtime_error("Vulkan SdpaImpl: not implemented");
}
void AttentionImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan AttentionImpl: not implemented");
}
void FlashAttentionImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan FlashAttentionImpl: not implemented");
}

template struct SoftmaxImpl<Device::VULKAN>;
template struct DiagMaskInfImpl<Device::VULKAN>;
template struct SdpaImpl<Device::VULKAN>;
template struct AttentionImpl<Device::VULKAN>;
template struct FlashAttentionImpl<Device::VULKAN>;

}
