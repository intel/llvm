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

#include <unified-runtime/ur_api.h>

#include <cstddef>
#include <string_view>

namespace sycl {
inline namespace _V1 {
namespace detail {

class context_impl;
class device_impl;
class DeviceKernelInfo;

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

// O(1) cached-kernel lookup + direct UR dispatch for kernel_device_specific
// info queries. Each helper fetches (or builds) the UR kernel via
// ProgramManager::getOrCreateKernel and then issues the corresponding
// urKernelGet*Info call, writing the raw result into `Result`. `ResultSize`
// must match the size of the UR-reported value (e.g. sizeof(size_t)*3 for a
// three-element range, sizeof(uint32_t) for a uint32 query, etc.). These
// bypass the sycl::kernel wrapper to keep the fast path allocation-free.

__SYCL_EXPORT void queryCachedKernelGroupInfo(context_impl &CtxImpl,
                                              device_impl &DevImpl,
                                              DeviceKernelInfo &KernelInfo,
                                              ur_kernel_group_info_t InfoCode,
                                              size_t ResultSize, void *Result);

__SYCL_EXPORT void queryCachedKernelSubGroupInfo(
    context_impl &CtxImpl, device_impl &DevImpl, DeviceKernelInfo &KernelInfo,
    ur_kernel_sub_group_info_t InfoCode, size_t ResultSize, void *Result);

__SYCL_EXPORT void queryCachedKernelUrKernelInfo(
    context_impl &CtxImpl, device_impl &DevImpl, DeviceKernelInfo &KernelInfo,
    ur_kernel_info_t InfoCode, size_t ResultSize, void *Result);

// ext_intel_spill_memory_size has custom per-device demultiplexing logic that
// doesn't fit the simple "one UR call with a buffer" pattern above.
__SYCL_EXPORT size_t queryCachedKernelSpillMemSize(
    context_impl &CtxImpl, device_impl &DevImpl, DeviceKernelInfo &KernelInfo);

} // namespace detail
} // namespace _V1
} // namespace sycl
