//==--------------------- get_device_kernel_info.hpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/compile_time_kernel_info.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/kernel_desc.hpp>

#include <memory>
#include <string_view>

namespace sycl {
inline namespace _V1 {
namespace detail {

class context_impl;
class device_impl;
class DeviceKernelInfo;
class kernel_impl;

// Lifetime of the underlying `DeviceKernelInfo` is tied to the availability of
// the `sycl_device_binaries` corresponding to this kernel. In other words, once
// user library is unloaded (see __sycl_unregister_lib), program manager
// destroys this `DeviceKernelInfo` object and the reference returned from here
// becomes stale.
__SYCL_EXPORT DeviceKernelInfo &
getDeviceKernelInfo(const CompileTimeKernelInfoTy &);

template <class Kernel> DeviceKernelInfo &getDeviceKernelInfo() {
  static DeviceKernelInfo &Info =
      getDeviceKernelInfo(CompileTimeKernelInfo<Kernel>);
  return Info;
}

// Overload for free function kernels
// Uses FreeFunctionInfoData which is specialized by the integration header
__SYCL_EXPORT DeviceKernelInfo &
getDeviceKernelInfo(std::string_view KernelName);

template <auto *Func> DeviceKernelInfo &getDeviceKernelInfo() {
  static DeviceKernelInfo &Info =
      getDeviceKernelInfo(FreeFunctionInfoData<Func>::getFunctionName());
  return Info;
}

// O(1) cached-kernel lookup. Uses ProgramManager::getOrCreateKernel to fetch
// (or build and cache) the UR kernel for this context/device/KernelInfo, and
// returns a kernel_impl wrapping it. Callers wrap into a sycl::kernel and use
// its existing get_info<Param>(device) entry points.
__SYCL_EXPORT std::shared_ptr<kernel_impl>
getCachedKernelImpl(context_impl &CtxImpl, device_impl &DevImpl,
                    DeviceKernelInfo &KernelInfo);

} // namespace detail
} // namespace _V1
} // namespace sycl
