#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include "utils/utils.hpp"

constexpr int32_t TENSOR_MAX_SRC = 5;
constexpr int32_t TENSOR_MAX_DIMS = 4;
constexpr int32_t TENSOR_MAX_OP_PARAMS = 4;

struct Tensor{
    // === Cache line 0 (64B): 内存定位 + 图拓扑 ===
    void* data = nullptr;       // CPU/CUDA: 指向实际数据的指针；Vulkan: nullptr（使用 device_handle）
    size_t device_handle = 0;   // Vulkan: VkBuffer handle；CPU/CUDA: 0
    size_t offset = 0;          // data 在内存池中的偏移（从基地址算起）
    std::array<Tensor*, TENSOR_MAX_SRC> src = {nullptr,nullptr,nullptr,nullptr,nullptr};// 源 tensor（用于计算图）

    // === Cache line 1: 形状 + 调度信息 ===
    std::array<int64_t, TENSOR_MAX_DIMS> dims = {0,0,0,0};      // 各维度大小,默认都是0
    enum OperationType op_type = OperationType::OP_TYPE_NONE;    // add,sub,mul,div.....
    enum Device device = Device::CPU;                            // 逻辑设备（CPU/CUDA/SYCL/VULKAN）
    enum DataType dtype = DataType::GGML_TYPE_F32;               // fp16,fp32,bf16,.....
    enum TensorType type = TensorType::TENSOR_TYPE_UNKNOWN;      // bias,input,activation...
    int layer_id = -1;                                           // 构建计算图时的层ID

    // === Cache line 2: 运行参数 ===
    std::array<float, TENSOR_MAX_OP_PARAMS> op_params = {0,0,0,0};     // 操作参数
    std::array<uint64_t, TENSOR_MAX_DIMS> strides = {0,0,0,0};         // 各维度字节跨度

    // === Cache line 3: 冷数据 ===
    std::string name;

    // ========== 工具方法 ==========
    size_t num_elements() const {
        size_t total = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && dims[i] != 0; ++i) {
            total *= std::abs(dims[i]);
        }
        return total;
    }
    size_t bytes() const {
        return num_elements() * data_type_size(dtype);
    }
    size_t bytes_at(int64_t resolve) const {
        size_t total = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && dims[i] != 0; ++i) {
            total *= (dims[i] < 0 ? resolve : dims[i]);
        }
        return total * data_type_size(dtype);
    }
    bool is_computed() const {
        return op_type != OperationType::OP_TYPE_NONE && type != TensorType::TENSOR_TYPE_WEIGHT && type != TensorType::TENSOR_TYPE_INPUT;
    }
};
