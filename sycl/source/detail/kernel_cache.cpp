//==-- kernel_cache.cpp - Fast kernel cache access implementation ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/kernel_cache.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/range.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Helper to get cached kernel handle
static ur_kernel_handle_t getCachedKernel(context_impl &CtxImpl,
                                          device_impl &DevImpl,
                                          DeviceKernelInfo &KernelInfo) {
  // Empty NDRDesc is fine for info queries
  NDRDescT NDRDesc{};

  // Use the fast kernel cache - O(1) lookup!
  FastKernelCacheValPtr KernelCacheVal =
      ProgramManager::getInstance().getOrCreateKernel(CtxImpl, DevImpl,
                                                      KernelInfo, NDRDesc);

  return KernelCacheVal->MKernelHandle;
}

size_t getKernelWorkGroupSize(context_impl &CtxImpl, device_impl &DevImpl,
                              DeviceKernelInfo &KernelInfo) {
  ur_kernel_handle_t Kernel = getCachedKernel(CtxImpl, DevImpl, KernelInfo);
  ur_device_handle_t Device = DevImpl.getHandleRef();

  size_t result;
  CtxImpl.getAdapter().call<UrApiKind::urKernelGetGroupInfo>(
      Kernel, Device, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE, sizeof(size_t),
      &result, nullptr);
  return result;
}

range<3> getKernelCompileWorkGroupSize(context_impl &CtxImpl,
                                       device_impl &DevImpl,
                                       DeviceKernelInfo &KernelInfo) {
  ur_kernel_handle_t Kernel = getCachedKernel(CtxImpl, DevImpl, KernelInfo);
  ur_device_handle_t Device = DevImpl.getHandleRef();

  size_t result[3];
  CtxImpl.getAdapter().call<UrApiKind::urKernelGetGroupInfo>(
      Kernel, Device, UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
      sizeof(result), result, nullptr);
  return range<3>(result[0], result[1], result[2]);
}

size_t getKernelPreferredWorkGroupSizeMultiple(context_impl &CtxImpl,
                                               device_impl &DevImpl,
                                               DeviceKernelInfo &KernelInfo) {
  ur_kernel_handle_t Kernel = getCachedKernel(CtxImpl, DevImpl, KernelInfo);
  ur_device_handle_t Device = DevImpl.getHandleRef();

  size_t result;
  CtxImpl.getAdapter().call<UrApiKind::urKernelGetGroupInfo>(
      Kernel, Device, UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
      sizeof(size_t), &result, nullptr);
  return result;
}

size_t getKernelPrivateMemSize(context_impl &CtxImpl, device_impl &DevImpl,
                               DeviceKernelInfo &KernelInfo) {
  ur_kernel_handle_t Kernel = getCachedKernel(CtxImpl, DevImpl, KernelInfo);
  ur_device_handle_t Device = DevImpl.getHandleRef();

  size_t result;
  CtxImpl.getAdapter().call<UrApiKind::urKernelGetGroupInfo>(
      Kernel, Device, UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE, sizeof(size_t),
      &result, nullptr);
  return result;
}

uint32_t getKernelMaxSubGroupSize(context_impl &CtxImpl, device_impl &DevImpl,
                                  DeviceKernelInfo &KernelInfo) {
  ur_kernel_handle_t Kernel = getCachedKernel(CtxImpl, DevImpl, KernelInfo);
  ur_device_handle_t Device = DevImpl.getHandleRef();

  uint32_t result;
  CtxImpl.getAdapter().call<UrApiKind::urKernelGetSubGroupInfo>(
      Kernel, Device, UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE,
      sizeof(uint32_t), &result, nullptr);
  return result;
}

uint32_t getKernelCompileNumSubGroups(context_impl &CtxImpl,
                                      device_impl &DevImpl,
                                      DeviceKernelInfo &KernelInfo) {
  ur_kernel_handle_t Kernel = getCachedKernel(CtxImpl, DevImpl, KernelInfo);
  ur_device_handle_t Device = DevImpl.getHandleRef();

  uint32_t result;
  CtxImpl.getAdapter().call<UrApiKind::urKernelGetSubGroupInfo>(
      Kernel, Device, UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS,
      sizeof(uint32_t), &result, nullptr);
  return result;
}

uint32_t getKernelCompileSubGroupSize(context_impl &CtxImpl,
                                      device_impl &DevImpl,
                                      DeviceKernelInfo &KernelInfo) {
  ur_kernel_handle_t Kernel = getCachedKernel(CtxImpl, DevImpl, KernelInfo);
  ur_device_handle_t Device = DevImpl.getHandleRef();

  uint32_t result;
  CtxImpl.getAdapter().call<UrApiKind::urKernelGetSubGroupInfo>(
      Kernel, Device, UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL,
      sizeof(uint32_t), &result, nullptr);
  return result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
