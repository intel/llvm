//==--------------------- get_device_kernel_info.hpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/compile_time_kernel_info.hpp>
#include <sycl/detail/kernel_desc.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

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

} // namespace detail
} // namespace _V1
} // namespace sycl
