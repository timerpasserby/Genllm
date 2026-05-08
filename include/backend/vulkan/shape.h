#pragma once
#include "core/tensor.hpp"

namespace ops {
    template <Device D> struct ReshapeImpl;
    template <Device D> struct PermuteImpl;
    template <Device D> struct ConcatImpl;
    template <Device D> struct RepeatImpl;

    template <> struct ReshapeImpl<Device::VULKAN>  { static void execute(Tensor* out); };
    template <> struct PermuteImpl<Device::VULKAN>  { static void execute(Tensor* out); };
    template <> struct ConcatImpl<Device::VULKAN>   { static void execute(Tensor* out); };
    template <> struct RepeatImpl<Device::VULKAN>   { static void execute(Tensor* out); };

    extern template struct ReshapeImpl<Device::VULKAN>;
    extern template struct PermuteImpl<Device::VULKAN>;
    extern template struct ConcatImpl<Device::VULKAN>;
    extern template struct RepeatImpl<Device::VULKAN>;
}
