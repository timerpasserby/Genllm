#pragma once
#include "core/tensor.hpp"

namespace ops {
    template <Device D> struct ApplyRopeImpl;

    template <> struct ApplyRopeImpl<Device::VULKAN> { static void execute(Tensor* out); };

    extern template struct ApplyRopeImpl<Device::VULKAN>;
}
