#include "utils/dtype_traits.hpp"
#include "backend/cuda/mamba.h"

namespace ops {

void CausalConv1dImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    // TODO: implement causal conv1d
    throw std::runtime_error("CausalConv1dImpl<CUDA>: not implemented");
}

void SsmScanImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    // TODO: implement ssm scan
    throw std::runtime_error("SsmScanImpl<CUDA>: not implemented");
}

template struct CausalConv1dImpl<Device::CUDA>;
template struct SsmScanImpl<Device::CUDA>;

}
