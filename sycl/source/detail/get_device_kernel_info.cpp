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
  // The wrapping kernel_impl keeps the FastKernelCacheVal alive via
  // shared_ptr rather than issuing a urKernelRetain/urKernelRelease pair.
  return std::make_shared<kernel_impl>(kernel_impl::for_cached_info_query_t{},
                                       std::move(KernelCacheVal), CtxImpl);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
