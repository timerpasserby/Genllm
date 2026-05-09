#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ApplyRopeImpl;

    template <>
    struct ApplyRopeImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };
    extern template struct ApplyRopeImpl<Device::CUDA>;



    
    // template <Device D> struct RopeCacheImpl; // 这个算子只要cpu端实现就可以
    // template <>
    // struct RopeCacheImpl<Device::CUDA> {
    //     static void execute(Tensor* out, int32_t dev_id);
    // };
    // extern template struct RopeCacheImpl<Device::CUDA>;
}
