

#ifdef BACKEND_VULKAN

#include "backend/vulkan/shape.h"
#include "utils/dtype_traits.hpp"

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/permute.h"
#include "backend/vulkan/push_constants.h"

namespace ops {

static void dispatch_permute(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out, const PermutePushConstants& pc)
{
    Tensor* x = out->src[0];
    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {x, out});

    uint32_t group_x = (static_cast<uint32_t>(pc.total) + 255) / 256;

    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);
    ctx.pushConstants(dev_id, &pc, sizeof(PermutePushConstants));
    ctx.dispatch(dev_id, group_x, 1, 1);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
}

void ReshapeImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    const Tensor* x = out->src[0];
    out->device_handle = x->device_handle;
    out->offset        = x->offset;

    size_t elem_sz = data_type_size(out->dtype);
    size_t stride  = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (out->dims[i] == 0) {
            out->strides[i] = 0;
        } else {
            out->strides[i] = stride * elem_sz;
            stride *= static_cast<size_t>(out->dims[i]);
        }
    }
}

void PermuteImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    const Tensor* x = out->src[0];

    int ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) ndim = i + 1;
    if (ndim <= 1) return;

    size_t elem_sz = data_type_size(out->dtype);

    PermutePushConstants pc{};
    pc.ndim  = ndim;
    pc.total = static_cast<int32_t>(out->num_elements());
    for (int i = 0; i < ndim; i++) {
        pc.out_strides[i] = static_cast<int32_t>(out->strides[i] / elem_sz);
        pc.src_strides[i] = static_cast<int32_t>(x->strides[i] / elem_sz);
        pc.perm[i]        = static_cast<int32_t>(out->op_params[i]);
    }

    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_permute(ctx, dev_id, "permute_f16", vkspv::permute_f16_spv, vkspv::permute_f16_spv_len, out, pc);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_permute(ctx, dev_id, "permute_bf16", vkspv::permute_bf16_spv, vkspv::permute_bf16_spv_len, out, pc);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_permute(ctx, dev_id, "permute_f32", vkspv::permute_f32_spv, vkspv::permute_f32_spv_len, out, pc);
        }
    });
}

void ConcatImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan ConcatImpl: not implemented");
}
void RepeatImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan RepeatImpl: not implemented");
}

void NarrowImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    const Tensor* x = out->src[0];
    int32_t dim   = static_cast<int32_t>(out->op_params[0]);
    int64_t start = static_cast<int64_t>(out->op_params[1]);
    int64_t size  = static_cast<int64_t>(out->op_params[2]);
    size_t elem_sz = data_type_size(out->dtype);

    int64_t outer = 1;
    for (int d = 0; d < dim; ++d)
        if (x->dims[d] > 0) outer *= x->dims[d];

    size_t src_block = 1;
    for (int d = dim; d < TENSOR_MAX_DIMS && x->dims[d] > 0; ++d)
        src_block *= x->dims[d];
    src_block *= elem_sz;

    size_t dst_block = size * elem_sz;
    size_t offset    = start * elem_sz;

    auto& ctx = VulkanContext::get();
    if (!ctx.isRecording(dev_id))
        throw std::runtime_error("NarrowImpl<VULKAN>: not in recording mode");
    vk::CommandBuffer cmd = ctx.cmdBuffer(dev_id);

    vk::Buffer src_buf = reinterpret_cast<VkBuffer>(x->device_handle);
    vk::Buffer dst_buf = reinterpret_cast<VkBuffer>(out->device_handle);

    std::vector<vk::BufferCopy> regions;
    for (int64_t i = 0; i < outer; ++i) {
        regions.push_back({
            x->offset + i * src_block + offset,
            out->offset + i * dst_block,
            dst_block
        });
    }
    if (!regions.empty()) cmd.copyBuffer(src_buf, dst_buf, regions);
}

template struct ReshapeImpl<Device::VULKAN>;
template struct PermuteImpl<Device::VULKAN>;
template struct ConcatImpl<Device::VULKAN>;
template struct RepeatImpl<Device::VULKAN>;
template struct NarrowImpl<Device::VULKAN>;

}

#endif
