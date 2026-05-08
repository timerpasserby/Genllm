#pragma once
#include "core/tensor.hpp"

namespace ops {
    template <Device D> struct RmsNormImpl;
    template <Device D> struct LayerNormImpl;

    template <> struct RmsNormImpl<Device::VULKAN>   { static void execute(Tensor* out); };
    template <> struct LayerNormImpl<Device::VULKAN>  { static void execute(Tensor* out); };

    extern template struct RmsNormImpl<Device::VULKAN>;
    extern template struct LayerNormImpl<Device::VULKAN>;
}
