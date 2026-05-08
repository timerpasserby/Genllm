#pragma once

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <format>
#include <print>
#include <cstddef>

class VulkanContext {
public:
    static VulkanContext& get(int device_id = 0) {
        static VulkanContext instance;
        return instance;
    }

    vk::Device device() const { return device_; }
    vk::PhysicalDevice physical_device() const { return physical_device_; }
    vk::Queue compute_queue() const { return compute_queue_; }
    uint32_t compute_queue_family() const { return compute_queue_family_; }

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    ~VulkanContext() {
        if (device_) {
            device_.waitIdle();
            for (auto& [name, info] : pipeline_infos_) {
                device_.destroyPipeline(info.pipeline);
                device_.destroyPipelineLayout(info.layout);
                device_.destroyDescriptorSetLayout(info.ds_layout);
            }
            device_.destroyDescriptorPool(descriptor_pool_);
            device_.destroyCommandPool(command_pool_);
            device_.destroy();
        }
        if (instance_) {
            instance_.destroy();
        }
    }

    // ==================== Pipeline 管理（懒加载） ====================

    struct PipelineInfo {
        vk::Pipeline pipeline;
        vk::PipelineLayout layout;
        vk::DescriptorSetLayout ds_layout;
        uint32_t binding_count;
    };

    const PipelineInfo& getOrCreatePipeline(
        const std::string& name,
        const uint32_t* spv_data,
        size_t spv_size_words,
        uint32_t binding_count,
        uint32_t push_constant_size)
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        auto it = pipeline_infos_.find(name);
        if (it != pipeline_infos_.end()) return it->second;

        // 1. Descriptor Set Layout: N 个 storage buffer binding
        std::vector<vk::DescriptorSetLayoutBinding> bindings(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            bindings[i] = vk::DescriptorSetLayoutBinding(
                i,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute);
        }
        vk::DescriptorSetLayoutCreateInfo ds_info({}, bindings);
        auto ds_layout = device_.createDescriptorSetLayout(ds_info);

        // 2. Push Constant Range
        vk::PushConstantRange pc_range(
            vk::ShaderStageFlagBits::eCompute, 0, push_constant_size);

        vk::PipelineLayoutCreateInfo layout_info(
            {}, ds_layout, pc_range);
        auto layout = device_.createPipelineLayout(layout_info);

        // 3. Shader Module
        vk::ShaderModuleCreateInfo shader_info(
            {}, spv_size_words * sizeof(uint32_t), spv_data);
        auto shader_module = device_.createShaderModule(shader_info);

        // 4. Compute Pipeline
        vk::PipelineShaderStageCreateInfo stage_info(
            {}, vk::ShaderStageFlagBits::eCompute, shader_module, "main");

        vk::ComputePipelineCreateInfo pipe_info({}, stage_info, layout);
        auto [result, pipeline] = device_.createComputePipeline(nullptr, pipe_info);

        device_.destroyShaderModule(shader_module);

        if (result != vk::Result::eSuccess) {
            device_.destroyDescriptorSetLayout(ds_layout);
            device_.destroyPipelineLayout(layout);
            throw std::runtime_error(std::format(
                "VulkanContext: createComputePipeline '{}' failed ({})", name, vk::to_string(result)));
        }

        // 5. 缓存
        auto [ins_it, ok] = pipeline_infos_.emplace(name, PipelineInfo{
            pipeline, layout, ds_layout, binding_count});
        return ins_it->second;
    }

    // ==================== Descriptor Set 分配 ====================

    vk::DescriptorSet allocateDescriptorSet(vk::DescriptorSetLayout layout) {
        vk::DescriptorSetAllocateInfo alloc_info(descriptor_pool_, 1, &layout);
        auto sets = device_.allocateDescriptorSets(alloc_info);
        if (sets.empty()) {
            throw std::runtime_error("VulkanContext: allocateDescriptorSet failed");
        }
        return sets[0];
    }

    void freeDescriptorSet(vk::DescriptorSet ds) {
        device_.freeDescriptorSets(descriptor_pool_, ds);
    }

    void updateDescriptorSets(
        vk::DescriptorSet ds,
        const std::vector<vk::DescriptorBufferInfo>& buffer_infos)
    {
        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(buffer_infos.size());
        for (uint32_t i = 0; i < buffer_infos.size(); ++i) {
            writes.emplace_back(
                ds, i, 0, 1, vk::DescriptorType::eStorageBuffer,
                nullptr, &buffer_infos[i]);
        }
        device_.updateDescriptorSets(writes, {});
    }

    // ==================== 命令提交 ====================

    vk::CommandBuffer beginCommandBuffer() {
        vk::CommandBufferAllocateInfo alloc_info(
            command_pool_, vk::CommandBufferLevel::ePrimary, 1);
        auto cmds = device_.allocateCommandBuffers(alloc_info);
        if (cmds.empty()) {
            throw std::runtime_error("VulkanContext: allocateCommandBuffer failed");
        }
        vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmds[0].begin(begin_info);
        return cmds[0];
    }

    void endSubmitAndWait(vk::CommandBuffer cmd) {
        cmd.end();

        vk::SubmitInfo submit_info({}, {}, cmd);
        vk::Fence fence = device_.createFence({});
        compute_queue_.submit(submit_info, fence);
        VkFence vk_fence = static_cast<VkFence>(fence);
        vkWaitForFences(static_cast<VkDevice>(device_), 1, &vk_fence, VK_TRUE, UINT64_MAX);
        device_.destroyFence(fence);
        device_.freeCommandBuffers(command_pool_, cmd);
    }

    // ==================== Buffer 辅助 ====================

    // 创建 staging buffer（host-visible），用于 host → device 拷贝
    vk::Buffer createStagingBuffer(size_t size, vk::DeviceMemory* out_memory, void** out_mapped) {
        vk::BufferCreateInfo buf_info(
            {}, size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive);
        auto buffer = device_.createBuffer(buf_info);

        auto mem_reqs = device_.getBufferMemoryRequirements(buffer);
        auto mem_props = physical_device_.getMemoryProperties();

        uint32_t mem_type = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((mem_reqs.memoryTypeBits & (1 << i)) &&
                (mem_props.memoryTypes[i].propertyFlags &
                 (vk::MemoryPropertyFlagBits::eHostVisible |
                  vk::MemoryPropertyFlagBits::eHostCoherent))) {
                mem_type = i;
                break;
            }
        }
        if (mem_type == UINT32_MAX) {
            device_.destroyBuffer(buffer);
            throw std::runtime_error("VulkanContext: no host-visible memory for staging");
        }

        vk::MemoryAllocateInfo alloc_info(mem_reqs.size, mem_type);
        auto memory = device_.allocateMemory(alloc_info);
        device_.bindBufferMemory(buffer, memory, 0);

        *out_memory = memory;
        *out_mapped = device_.mapMemory(memory, 0, size);
        return buffer;
    }

    void destroyStagingBuffer(vk::Buffer buffer, vk::DeviceMemory memory, void* mapped) {
        if (mapped)  device_.unmapMemory(memory);
        if (buffer)  device_.destroyBuffer(buffer);
        if (memory)  device_.freeMemory(memory);
    }

private:
    VulkanContext() { init(); }

    void init() {
        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        createDescriptorPool();
    }

    void createInstance() {
        vk::ApplicationInfo app_info("Genllm", 1, "GenllmEngine", 1, VK_API_VERSION_1_3);
        vk::InstanceCreateInfo create_info({}, &app_info);
        auto result = vk::createInstance(&create_info, nullptr, &instance_);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format(
                "VulkanContext: createInstance failed ({})", vk::to_string(result)));
        }
    }

    void pickPhysicalDevice() {
        auto devices = instance_.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("VulkanContext: no Vulkan-capable devices");
        }

        int best_score = -1;
        for (auto& dev : devices) {
            auto props = dev.getProperties();
            int score = 0;
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) score = 10;
            else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) score = 5;

            auto families = dev.getQueueFamilyProperties();
            bool has_compute = false;
            for (uint32_t i = 0; i < families.size(); ++i) {
                if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                    has_compute = true;
                    break;
                }
            }
            if (!has_compute) continue;

            if (score > best_score) {
                best_score = score;
                physical_device_ = dev;
            }
        }
        if (!physical_device_) {
            throw std::runtime_error("VulkanContext: no suitable GPU found");
        }
        auto props = physical_device_.getProperties();
        std::println("Vulkan: selected GPU: {}", std::string_view(props.deviceName));
    }

    void createLogicalDevice() {
        auto families = physical_device_.getQueueFamilyProperties();
        compute_queue_family_ = 0;
        for (uint32_t i = 0; i < families.size(); ++i) {
            if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                compute_queue_family_ = i;
                break;
            }
        }

        float priority = 1.0f;
        vk::DeviceQueueCreateInfo queue_info({}, compute_queue_family_, 1, &priority);
        vk::DeviceCreateInfo device_info({}, queue_info);
        auto result = physical_device_.createDevice(&device_info, nullptr, &device_);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(std::format(
                "VulkanContext: createDevice failed ({})", vk::to_string(result)));
        }
        compute_queue_ = device_.getQueue(compute_queue_family_, 0);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo pool_info(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, compute_queue_family_);
        command_pool_ = device_.createCommandPool(pool_info);
    }

    void createDescriptorPool() {
        vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer, 4096);
        vk::DescriptorPoolCreateInfo pool_info(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1024, pool_size);
        descriptor_pool_ = device_.createDescriptorPool(pool_info);
    }

    // Vulkan 核心
    vk::Instance instance_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::Queue compute_queue_;
    uint32_t compute_queue_family_ = 0;
    vk::CommandPool command_pool_;
    vk::DescriptorPool descriptor_pool_;

    // Pipeline 缓存
    std::mutex pipeline_mutex_;
    std::unordered_map<std::string, PipelineInfo> pipeline_infos_;
};

#endif // BACKEND_VULKAN
