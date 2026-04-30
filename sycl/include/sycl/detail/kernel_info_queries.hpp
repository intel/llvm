//==-- kernel_info_queries.hpp - Kernel info queries using cached kernels --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>
#include <sycl/detail/get_device_kernel_info.hpp>

#include <cstddef>
#include <cstdint>

namespace sycl {
inline namespace _V1 {

// Forward declarations
class context;
class device;
template <int Dimensions> class range;

namespace detail {

// Forward declarations
class context_impl;
class device_impl;

// Fast O(1) kernel cache lookup and info query
// These functions use getOrCreateKernel for O(1) lookup and then delegate
// to the existing helpers in source/detail/kernel_info.hpp for the actual
// UR API calls (avoiding code duplication)

__SYCL_EXPORT size_t getKernelWorkGroupSize(context_impl &CtxImpl,
                                            device_impl &DevImpl,
                                            DeviceKernelInfo &KernelInfo);

__SYCL_EXPORT range<3>
getKernelCompileWorkGroupSize(context_impl &CtxImpl, device_impl &DevImpl,
                              DeviceKernelInfo &KernelInfo);

__SYCL_EXPORT size_t getKernelPreferredWorkGroupSizeMultiple(
    context_impl &CtxImpl, device_impl &DevImpl, DeviceKernelInfo &KernelInfo);

__SYCL_EXPORT size_t getKernelPrivateMemSize(context_impl &CtxImpl,
                                             device_impl &DevImpl,
                                             DeviceKernelInfo &KernelInfo);

__SYCL_EXPORT uint32_t getKernelMaxSubGroupSize(context_impl &CtxImpl,
                                                device_impl &DevImpl,
                                                DeviceKernelInfo &KernelInfo);

__SYCL_EXPORT uint32_t getKernelCompileNumSubGroups(
    context_impl &CtxImpl, device_impl &DevImpl, DeviceKernelInfo &KernelInfo);

__SYCL_EXPORT uint32_t getKernelCompileSubGroupSize(
    context_impl &CtxImpl, device_impl &DevImpl, DeviceKernelInfo &KernelInfo);

} // namespace detail
} // namespace _V1
} // namespace sycl
