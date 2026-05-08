#include "backend/vulkan/memcpy.h"
#include <stdexcept>

namespace ops {

void MemcpyImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan MemcpyImpl: not implemented");
}

template struct MemcpyImpl<Device::VULKAN>;

}
