//==-------------------- get_device_kernel_info.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/get_device_kernel_info.hpp>

#include <detail/program_manager/program_manager.hpp>

#include <string_view>

namespace sycl {
inline namespace _V1 {
namespace detail {

__SYCL_EXPORT DeviceKernelInfo &
getDeviceKernelInfo(const CompileTimeKernelInfoTy &Info,
                    const void *CallerAnchor) {
  return ProgramManager::getInstance().getDeviceKernelInfo(Info, CallerAnchor);
}

// Kept as a separate exported symbol for binary compatibility with user
// libraries built against older SYCL headers that didn't pass a caller
// anchor. Such callers don't participate in per-DSO disambiguation and will
// fall back to compile-time-info matching.
__SYCL_EXPORT DeviceKernelInfo &
getDeviceKernelInfo(const CompileTimeKernelInfoTy &Info) {
  return ProgramManager::getInstance().getDeviceKernelInfo(Info, nullptr);
}

DeviceKernelInfo &getDeviceKernelInfo(std::string_view KernelName) {
  return ProgramManager::getInstance().getDeviceKernelInfo(KernelName);
}

DeviceKernelInfo &getDeviceKernelInfo(std::string_view KernelName,
                                      const void *CallerAnchor) {
  return ProgramManager::getInstance().getDeviceKernelInfo(KernelName,
                                                           CallerAnchor);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
