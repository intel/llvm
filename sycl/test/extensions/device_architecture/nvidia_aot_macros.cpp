// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_88__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_88 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_90A__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_90a %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_100__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_100 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_100A__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_100 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_100F__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_100 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_101__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_101 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_101A__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_101 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_101F__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_101 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_103__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_103 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_103A__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_103 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_103F__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_103 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_110__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_110 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_110A__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_110 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_110F__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_110 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_120__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_120 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_120A__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_120 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_120F__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_120 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_121__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_121 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_121A__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_121 %s
// RUN: %clangxx -fsycl -fsyntax-only -D__SYCL_TARGET_NVIDIA_GPU_SM_121F__=1 -DEXPECTED_ARCH=nvidia_gpu_sm_121 %s

#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr auto CurrentArchitecture =
    sycl::detail::get_current_architecture_aot();

static_assert(sycl::detail::is_allowable_aot_mode);
static_assert(CurrentArchitecture.has_value());
static_assert(*CurrentArchitecture == syclex::architecture::EXPECTED_ARCH);
static_assert(syclex::architecture::nvidia_gpu_sm_101 ==
              syclex::architecture::nvidia_gpu_sm_110);
