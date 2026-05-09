#include "backend/vulkan/arithmetic.h"

#ifdef BACKEND_VULKAN

#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/add_f32.h"
#include "backend/vulkan/spv/add_bf16.h"
#include "backend/vulkan/spv/sub_f32.h"
#include "backend/vulkan/spv/sub_bf16.h"
#include "backend/vulkan/spv/mul_f32.h"
#include "backend/vulkan/spv/mul_bf16.h"
#include "backend/vulkan/spv/div_f32.h"
#include "backend/vulkan/spv/div_bf16.h"
#include <vulkan/vulkan.hpp>

namespace ops {

static void dispatch_binop(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 3, sizeof(uint32_t));

    Tensor* src0 = out->src[0];
    Tensor* src1 = out->src[1];

    vk::Buffer buf0 = reinterpret_cast<VkBuffer>(src0->device_handle);
    vk::Buffer buf1 = reinterpret_cast<VkBuffer>(src1->device_handle);
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);

    vk::DescriptorBufferInfo src0_info(buf0, src0->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo src1_info(buf1, src1->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {src0_info, src1_info, dst_info});

    uint32_t total = static_cast<uint32_t>(out->num_elements());

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &total);
    cmd.dispatch((total + 255) / 256, 1, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

static void dispatch_binop_dtype(
    VulkanContext& ctx, int dev_id, DataType dtype,
    const char* name_f32, const uint32_t* spv_f32, size_t len_f32,
    const char* name_bf16, const uint32_t* spv_bf16, size_t len_bf16,
    Tensor* out)
{
    switch (dtype) {
    case DataType::GGML_TYPE_F32:
        dispatch_binop(ctx, dev_id, name_f32, spv_f32, len_f32, out);
        break;
    case DataType::GGML_TYPE_BF16:
        dispatch_binop(ctx, dev_id, name_bf16, spv_bf16, len_bf16, out);
        break;
    default:
        throw std::runtime_error(std::format(
            "Vulkan arithmetic: unsupported dtype {}", data_type_to_string(dtype)));
    }
}

void AddImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dispatch_binop_dtype(ctx, dev_id, out->dtype,
        "add_f32", vkspv::add_f32_spv, vkspv::add_f32_spv_len,
        "add_bf16", vkspv::add_bf16_spv, vkspv::add_bf16_spv_len, out);
}

void SubImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dispatch_binop_dtype(ctx, dev_id, out->dtype,
        "sub_f32", vkspv::sub_f32_spv, vkspv::sub_f32_spv_len,
        "sub_bf16", vkspv::sub_bf16_spv, vkspv::sub_bf16_spv_len, out);
}

void MulImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dispatch_binop_dtype(ctx, dev_id, out->dtype,
        "mul_f32", vkspv::mul_f32_spv, vkspv::mul_f32_spv_len,
        "mul_bf16", vkspv::mul_bf16_spv, vkspv::mul_bf16_spv_len, out);
}

void DivImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dispatch_binop_dtype(ctx, dev_id, out->dtype,
        "div_f32", vkspv::div_f32_spv, vkspv::div_f32_spv_len,
        "div_bf16", vkspv::div_bf16_spv, vkspv::div_bf16_spv_len, out);
}

template struct AddImpl<Device::VULKAN>;
template struct SubImpl<Device::VULKAN>;
template struct MulImpl<Device::VULKAN>;
template struct DivImpl<Device::VULKAN>;

}

#endif
