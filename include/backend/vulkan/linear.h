#pragma once
#include "core/tensor.hpp"

namespace ops {
    template <Device D> struct MatmulImpl;
    template <Device D> struct LinearImpl;
    template <Device D> struct TransposeImpl;

    template <> struct MatmulImpl<Device::VULKAN>    { static void execute(Tensor* out, int32_t dev_id); };
    template <> struct LinearImpl<Device::VULKAN>     { static void execute(Tensor* out, int32_t dev_id); };
    template <> struct TransposeImpl<Device::VULKAN>  { static void execute(Tensor* out, int32_t dev_id); };

    extern template struct MatmulImpl<Device::VULKAN>;
    extern template struct LinearImpl<Device::VULKAN>;
    extern template struct TransposeImpl<Device::VULKAN>;
}
