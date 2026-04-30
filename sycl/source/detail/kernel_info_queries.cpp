//==-- kernel_info_queries.cpp - Kernel info query implementation ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/kernel_info_queries.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_info.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Helper to get cached kernel and call the existing info query helper
// This avoids duplicating the UR API calls that are already in kernel_info.hpp
template <typename Param>
typename Param::return_type
queryCachedKernelInfo(context_impl &CtxImpl, device_impl &DevImpl,
                      DeviceKernelInfo &KernelInfo) {
  // Empty NDRDesc is fine for info queries
  NDRDescT NDRDesc{};

  // Use the fast kernel cache - O(1) lookup!
  FastKernelCacheValPtr KernelCacheVal =
      ProgramManager::getInstance().getOrCreateKernel(CtxImpl, DevImpl,
                                                      KernelInfo, NDRDesc);

  // Reuse existing helper from kernel_info.hpp - no code duplication!
  return get_kernel_device_specific_info<Param>(KernelCacheVal->MKernelHandle,
                                                DevImpl.getHandleRef(),
                                                CtxImpl.getAdapter());
}

size_t getKernelWorkGroupSize(context_impl &CtxImpl, device_impl &DevImpl,
                              DeviceKernelInfo &KernelInfo) {
  return queryCachedKernelInfo<info::kernel_device_specific::work_group_size>(
      CtxImpl, DevImpl, KernelInfo);
}

range<3> getKernelCompileWorkGroupSize(context_impl &CtxImpl,
                                       device_impl &DevImpl,
                                       DeviceKernelInfo &KernelInfo) {
  return queryCachedKernelInfo<
      info::kernel_device_specific::compile_work_group_size>(CtxImpl, DevImpl,
                                                             KernelInfo);
}

size_t getKernelPreferredWorkGroupSizeMultiple(context_impl &CtxImpl,
                                               device_impl &DevImpl,
                                               DeviceKernelInfo &KernelInfo) {
  return queryCachedKernelInfo<
      info::kernel_device_specific::preferred_work_group_size_multiple>(
      CtxImpl, DevImpl, KernelInfo);
}

size_t getKernelPrivateMemSize(context_impl &CtxImpl, device_impl &DevImpl,
                               DeviceKernelInfo &KernelInfo) {
  return queryCachedKernelInfo<info::kernel_device_specific::private_mem_size>(
      CtxImpl, DevImpl, KernelInfo);
}

uint32_t getKernelMaxSubGroupSize(context_impl &CtxImpl, device_impl &DevImpl,
                                  DeviceKernelInfo &KernelInfo) {
  return queryCachedKernelInfo<
      info::kernel_device_specific::max_sub_group_size>(CtxImpl, DevImpl,
                                                        KernelInfo);
}

uint32_t getKernelCompileNumSubGroups(context_impl &CtxImpl,
                                      device_impl &DevImpl,
                                      DeviceKernelInfo &KernelInfo) {
  return queryCachedKernelInfo<
      info::kernel_device_specific::compile_num_sub_groups>(CtxImpl, DevImpl,
                                                            KernelInfo);
}

uint32_t getKernelCompileSubGroupSize(context_impl &CtxImpl,
                                      device_impl &DevImpl,
                                      DeviceKernelInfo &KernelInfo) {
  return queryCachedKernelInfo<
      info::kernel_device_specific::compile_sub_group_size>(CtxImpl, DevImpl,
                                                            KernelInfo);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
