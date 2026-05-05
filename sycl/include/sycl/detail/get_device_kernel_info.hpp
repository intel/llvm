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
#include <sycl/info/info_desc.hpp>

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

// O(1) cached-kernel lookup + device-specific info query. Fetches the kernel
// via ProgramManager::getOrCreateKernel and dispatches to the existing
// get_kernel_device_specific_info<Param> helper in the library. Validation
// that kernel_impl::get_info would normally perform is replicated inside this
// function (see validateDeviceSpecificQuery in get_device_kernel_info.cpp).
// The 12 template instantiations below are the ABI boundary; callers in
// public headers (get_kernel_info.hpp) dispatch to them via the extern
// template declarations.
template <typename Param>
__SYCL_EXPORT typename Param::return_type
get_kernel_info_impl(context_impl &CtxImpl, device_impl &DevImpl,
                     DeviceKernelInfo &KernelInfo);

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  extern template __SYCL_EXPORT ReturnT                                        \
  get_kernel_info_impl<info::DescType::Desc>(context_impl &, device_impl &,    \
                                             DeviceKernelInfo &);
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  extern template __SYCL_EXPORT ReturnT                                        \
  get_kernel_info_impl<Namespace::info::DescType::Desc>(                       \
      context_impl &, device_impl &, DeviceKernelInfo &);
#include <sycl/info/ext_intel_kernel_info_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace _V1
} // namespace sycl
