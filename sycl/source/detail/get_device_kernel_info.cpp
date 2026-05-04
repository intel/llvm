//==-------------------- get_device_kernel_info.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/get_device_kernel_info.hpp>
#include <sycl/detail/kernel_desc.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <memory>
#include <string_view>

namespace sycl {
inline namespace _V1 {
namespace detail {

DeviceKernelInfo &getDeviceKernelInfo(const CompileTimeKernelInfoTy &Info) {
  return ProgramManager::getInstance().getDeviceKernelInfo(Info);
}

DeviceKernelInfo &getDeviceKernelInfo(std::string_view KernelName) {
  return ProgramManager::getInstance().getDeviceKernelInfo(KernelName);
}

std::shared_ptr<kernel_impl>
getCachedKernelImpl(context_impl &CtxImpl, device_impl &DevImpl,
                    DeviceKernelInfo &KernelInfo) {
  NDRDescT NDRDesc{};
  FastKernelCacheValPtr KernelCacheVal =
      ProgramManager::getInstance().getOrCreateKernel(CtxImpl, DevImpl,
                                                      KernelInfo, NDRDesc);
  // Retain the UR kernel so the wrapping kernel_impl owns an independent
  // reference (the FastKernelCacheVal also owns one).
  Managed<ur_kernel_handle_t> KernelHandle =
      KernelCacheVal->MKernelHandle.retain();
  return std::make_shared<kernel_impl>(std::move(KernelHandle), CtxImpl,
                                       /*KernelBundleImpl=*/nullptr,
                                       KernelCacheVal->MKernelArgMask);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
