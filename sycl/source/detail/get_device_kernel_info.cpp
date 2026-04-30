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
#include <detail/kernel_info.hpp>
#include <detail/program_manager/program_manager.hpp>

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

template <typename Param>
typename Param::return_type
queryCachedKernelInfo(context_impl &CtxImpl, device_impl &DevImpl,
                      DeviceKernelInfo &KernelInfo) {
  NDRDescT NDRDesc{};
  FastKernelCacheValPtr KernelCacheVal =
      ProgramManager::getInstance().getOrCreateKernel(CtxImpl, DevImpl,
                                                      KernelInfo, NDRDesc);
  return get_kernel_device_specific_info<Param>(KernelCacheVal->MKernelHandle,
                                                DevImpl.getHandleRef(),
                                                CtxImpl.getAdapter());
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, UrCode)              \
  template __SYCL_EXPORT ReturnT queryCachedKernelInfo<info::DescType::Desc>(  \
      context_impl &, device_impl &, DeviceKernelInfo &);
#include <sycl/info/kernel_device_specific_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(Namespace, DescType, Desc, ReturnT, UrCode)   \
  template __SYCL_EXPORT ReturnT                                               \
  queryCachedKernelInfo<Namespace::info::DescType::Desc>(                      \
      context_impl &, device_impl &, DeviceKernelInfo &);
#include <sycl/info/ext_intel_kernel_info_traits.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail
} // namespace _V1
} // namespace sycl
