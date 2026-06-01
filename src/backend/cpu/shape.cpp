#include <cstring>

#include <stdexcept>
#include "backend/cpu/shape.h"
#include "utils/dtype_traits.hpp"

// ────────────────────────────────────────────────────────────────
//  permute: 按置换表重排数据
// ────────────────────────────────────────────────────────────────
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void permute(
    T* out,
    const T* in,
    const int64_t* out_dims,
    const size_t* elem_strides,
    const int* perm,
    int ndim,
    size_t total
){
    size_t inner_dim = static_cast<size_t>(out_dims[ndim - 1]);
    if (perm[ndim - 1] == ndim - 1) {
        // 快速路径: 最内维未被置换，按行拷贝
        size_t outer_count = total / inner_dim;
        for (size_t r = 0; r < outer_count; ++r) {
            size_t remaining = r;
            size_t src_offset = 0;
            for (int d = ndim - 2; d >= 0; --d) {
                size_t dim_sz = static_cast<size_t>(out_dims[d]);
                size_t coord = remaining % dim_sz;
                remaining /= dim_sz;
                src_offset += coord * elem_strides[perm[d]];
            }
            for (size_t i = 0; i < inner_dim; ++i)
                out[r * inner_dim + i] = in[src_offset + i];
        }
    } else {
        // 慢速路径: 逐元素拷贝
        for (size_t idx = 0; idx < total; ++idx) {
            size_t remaining = idx;
            size_t src_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                size_t dim_sz = static_cast<size_t>(out_dims[d]);
                size_t coord = remaining % dim_sz;
                remaining /= dim_sz;
                src_offset += coord * elem_strides[perm[d]];
            }
            out[idx] = in[src_offset];
        }
    }
}

namespace ops {

    void ReshapeImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        out->data   = x->data;
        out->offset = x->offset;
        size_t stride = 1;
        for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
            if (out->dims[i] == 0) {
                out->strides[i] = 0;
            } else {
                out->strides[i] = stride * data_type_size(out->dtype);
                stride *= static_cast<size_t>(out->dims[i]);
            }
        }
    }

    void PermuteImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
       const Tensor* x = out->src[0];
        // 有效维度数
        int ndim = 0;
        for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) {
            ndim = i + 1;
        }
        // Read perm from op_params (stored during graph build)
        int perm[TENSOR_MAX_DIMS];
        for (int i = 0; i < ndim; ++i) {
            perm[i] = static_cast<int>(out->op_params[i]);
        }
        size_t elem_sz  = data_type_size(out->dtype);
        size_t total    = out->num_elements();
        auto* dst = static_cast<uint8_t*>(out->data);
        auto* src = static_cast<const uint8_t*>(x->data);
        size_t inner_dim   = static_cast<size_t>(out->dims[ndim - 1]);
        size_t inner_bytes = inner_dim * elem_sz;
        if (perm[ndim - 1] == ndim - 1) {
            // ── 快速路径: 最内维未被置换，源和输出都是行内连续 ──
            // 按外层维度遍历，每次拷贝一整行（inner_dim 个元素）
            size_t outer_count = total / inner_dim;
            for (size_t r = 0; r < outer_count; ++r) {
                // 将外层线性索引分解为多维坐标（不含最内维）
                size_t remaining  = r;
                size_t src_offset = 0;
                for (int d = ndim - 2; d >= 0; --d) {
                    size_t dim_sz = static_cast<size_t>(out->dims[d]);
                    size_t coord  = remaining % dim_sz;
                    remaining /= dim_sz;
                    src_offset += coord * x->strides[perm[d]];
                }
                std::memcpy(dst + r * inner_bytes, src + src_offset, inner_bytes);
            }
        } else {
            // ── 慢速路径: 最内维被置换，退化为逐元素拷贝 ──
            for (size_t idx = 0; idx < total; ++idx) {
                size_t remaining  = idx;
                size_t src_offset = 0;
                for (int d = ndim - 1; d >= 0; --d) {
                    size_t dim_sz = static_cast<size_t>(out->dims[d]);
                    size_t coord  = remaining % dim_sz;
                    remaining /= dim_sz;
                    src_offset += coord * x->strides[perm[d]];
                }
                std::memcpy(dst + idx * elem_sz, src + src_offset, elem_sz);
            }
        }
    }

    void ConcatImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        throw std::runtime_error("cpu::concat not implemented");
    }

    void RepeatImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        throw std::runtime_error("cpu::repeat not implemented");
    }

    void NarrowImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        int32_t dim   = static_cast<int32_t>(out->op_params[0]);
        int64_t start = static_cast<int64_t>(out->op_params[1]);
        int64_t size  = static_cast<int64_t>(out->op_params[2]);
        size_t elem_sz = data_type_size(out->dtype);

        // 计算窄维度之前的总元素数（batch 维度的乘积）
        int64_t outer = 1;
        for (int d = 0; d < dim; ++d)
            if (x->dims[d] > 0) outer *= x->dims[d];

        // 窄维度及其之后每个 block 的字节数
        size_t src_block_bytes = 1;
        for (int d = dim; d < TENSOR_MAX_DIMS && x->dims[d] > 0; ++d)
            src_block_bytes *= x->dims[d];
        src_block_bytes *= elem_sz;

        size_t dst_block_bytes = size * elem_sz;
        size_t copy_offset     = start * elem_sz;

        auto* dst = static_cast<uint8_t*>(out->data);
        auto* src = static_cast<const uint8_t*>(x->data);

        for (int64_t i = 0; i < outer; ++i) {
            std::memcpy(
                dst + i * dst_block_bytes,
                src + i * src_block_bytes + copy_offset,
                dst_block_bytes
            );
        }
    }

template struct ReshapeImpl<Device::CPU>;
template struct PermuteImpl<Device::CPU>;
template struct ConcatImpl<Device::CPU>;
template struct RepeatImpl<Device::CPU>;
template struct NarrowImpl<Device::CPU>;
}
