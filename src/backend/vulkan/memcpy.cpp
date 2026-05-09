#include "backend/vulkan/memcpy.h"
#include <stdexcept>

namespace ops {

void MemcpyImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    throw std::runtime_error("Vulkan MemcpyImpl: not implemented");
}

template struct MemcpyImpl<Device::VULKAN>;

}
