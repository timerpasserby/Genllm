#pragma once
#include "core/tensor.hpp"

namespace ops {
    template <Device D> struct SoftmaxImpl;
    template <Device D> struct DiagMaskInfImpl;
    template <Device D> struct SdpaImpl;
    template <Device D> struct AttentionImpl;
    template <Device D> struct FlashAttentionImpl;

    template <> struct SoftmaxImpl<Device::VULKAN>        { static void execute(Tensor* out); };
    template <> struct DiagMaskInfImpl<Device::VULKAN>    { static void execute(Tensor* out); };
    template <> struct SdpaImpl<Device::VULKAN>            { static void execute(Tensor* out, int32_t dev_id); };
    template <> struct AttentionImpl<Device::VULKAN>       { static void execute(Tensor* out); };
    template <> struct FlashAttentionImpl<Device::VULKAN>  { static void execute(Tensor* out); };

    extern template struct SoftmaxImpl<Device::VULKAN>;
    extern template struct DiagMaskInfImpl<Device::VULKAN>;
    extern template struct SdpaImpl<Device::VULKAN>;
    extern template struct AttentionImpl<Device::VULKAN>;
    extern template struct FlashAttentionImpl<Device::VULKAN>;
}
