#pragma once
#include "core/tensor.hpp"

namespace ops {
    template <Device D> struct ApplyRopeImpl;

    template <> struct ApplyRopeImpl<Device::VULKAN> { static void execute(Tensor* out, int32_t dev_id); };

    extern template struct ApplyRopeImpl<Device::VULKAN>;
}
