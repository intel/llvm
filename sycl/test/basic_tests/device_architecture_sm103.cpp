// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_103__=1 %s

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

namespace exp = sycl::ext::oneapi::experimental;
static_assert(sycl::detail::get_current_architecture_aot() ==
              exp::architecture::nvidia_gpu_sm_103);
static_assert(sycl::detail::get_device_architecture_category(
                  exp::architecture::nvidia_gpu_sm_103) ==
              exp::arch_category::nvidia_gpu);
