//==-------------------- get_device_kernel_info.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/kernel_desc.hpp>

#include <detail/global_handler.hpp>
#include <detail/program_manager/program_manager.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
KernelNameBasedCacheT *createKernelNameBasedCache() {
  return GlobalHandler::instance().createKernelNameBasedCache();
}
#endif

DeviceKernelInfo &getDeviceKernelInfo(const CompileTimeKernelInfoTy &Info) {
  return ProgramManager::getInstance().getOrCreateDeviceKernelInfo(Info);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
