#pragma once
#include "core/tensor.hpp"

namespace ops {
    template <Device D> struct AddImpl;
    template <Device D> struct SubImpl;
    template <Device D> struct MulImpl;
    template <Device D> struct DivImpl;

    template <> struct AddImpl<Device::VULKAN> { static void execute(Tensor* out); };
    template <> struct SubImpl<Device::VULKAN> { static void execute(Tensor* out); };
    template <> struct MulImpl<Device::VULKAN> { static void execute(Tensor* out); };
    template <> struct DivImpl<Device::VULKAN> { static void execute(Tensor* out); };

    extern template struct AddImpl<Device::VULKAN>;
    extern template struct SubImpl<Device::VULKAN>;
    extern template struct MulImpl<Device::VULKAN>;
    extern template struct DivImpl<Device::VULKAN>;
}
