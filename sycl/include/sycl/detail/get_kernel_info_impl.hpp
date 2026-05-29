//==------------------- get_kernel_info_impl.hpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/export.hpp>
#include <sycl/info/info_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

class context_impl;
class device_impl;
class DeviceKernelInfo;

// Fetches the cached kernel via the program manager and dispatches to
// get_kernel_device_specific_info. Validation that kernel_impl::get_info
// performs is replicated in the library-side definition (see
// validateDeviceSpecificQuery in source/detail/get_kernel_info_impl.cpp).
// Explicit instantiations for every kernel_device_specific descriptor form
// the ABI boundary; user code reaches these through the public entry points
// in ext/oneapi/get_kernel_info.hpp.
template <typename Param>
__SYCL_EXPORT typename Param::return_type
get_kernel_info_impl(context_impl &CtxImpl, device_impl &DevImpl,
                     DeviceKernelInfo &KernelInfo);

} // namespace detail
} // namespace _V1
} // namespace sycl
