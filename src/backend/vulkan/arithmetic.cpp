#include "backend/vulkan/arithmetic.h"

#ifdef BACKEND_VULKAN

#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/add_f32.h"
#include <vulkan/vulkan.hpp>

namespace ops {

void AddImpl<Device::VULKAN>::execute(Tensor* out) {
    auto& ctx = VulkanContext::get();
    auto& pipe = ctx.getOrCreatePipeline(
        "add_f32", vkspv::add_f32_spv, vkspv::add_f32_spv_len,
        3, sizeof(uint32_t));

    Tensor* src0 = out->src[0];
    Tensor* src1 = out->src[1];

    vk::Buffer buf0 = reinterpret_cast<VkBuffer>(src0->device_handle);
    vk::Buffer buf1 = reinterpret_cast<VkBuffer>(src1->device_handle);
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(pipe.ds_layout);

    vk::DescriptorBufferInfo src0_info(buf0, src0->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo src1_info(buf1, src1->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(ds, {src0_info, src1_info, dst_info});

    uint32_t total = static_cast<uint32_t>(out->num_elements());

    auto cmd = ctx.beginCommandBuffer();
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &total);
    cmd.dispatch((total + 255) / 256, 1, 1);
    ctx.endSubmitAndWait(cmd);

    ctx.freeDescriptorSet(ds);
}

void SubImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan SubImpl: not implemented");
}
void MulImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan MulImpl: not implemented");
}
void DivImpl<Device::VULKAN>::execute(Tensor* out) {
    throw std::runtime_error("Vulkan DivImpl: not implemented");
}

template struct AddImpl<Device::VULKAN>;
template struct SubImpl<Device::VULKAN>;
template struct MulImpl<Device::VULKAN>;
template struct DivImpl<Device::VULKAN>;

}

#endif
