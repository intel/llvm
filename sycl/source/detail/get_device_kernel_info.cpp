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
#include <detail/error_handling/error_handling.hpp>
#include <detail/global_handler.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <cstddef>
#include <string_view>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

DeviceKernelInfo &getDeviceKernelInfo(const CompileTimeKernelInfoTy &Info) {
  return ProgramManager::getInstance().getDeviceKernelInfo(Info);
}

DeviceKernelInfo &getDeviceKernelInfo(std::string_view KernelName) {
  return ProgramManager::getInstance().getDeviceKernelInfo(KernelName);
}

static FastKernelCacheValPtr fetchCachedKernel(context_impl &CtxImpl,
                                               device_impl &DevImpl,
                                               DeviceKernelInfo &KernelInfo) {
  NDRDescT NDRDesc{};
  return ProgramManager::getInstance().getOrCreateKernel(CtxImpl, DevImpl,
                                                         KernelInfo, NDRDesc);
}

void queryCachedKernelGroupInfo(context_impl &CtxImpl, device_impl &DevImpl,
                                DeviceKernelInfo &KernelInfo,
                                ur_kernel_group_info_t InfoCode,
                                size_t ResultSize, void *Result) {
  FastKernelCacheValPtr CacheVal =
      fetchCachedKernel(CtxImpl, DevImpl, KernelInfo);
  adapter_impl &Adapter = CtxImpl.getAdapter();
  ur_result_t Err = Adapter.call_nocheck<UrApiKind::urKernelGetGroupInfo>(
      CacheVal->MKernelHandle, DevImpl.getHandleRef(), InfoCode, ResultSize,
      Result, nullptr);
  if (Err != UR_RESULT_SUCCESS)
    kernel_get_group_info::handleErrorOrWarning(Err, InfoCode, Adapter);
}

void queryCachedKernelSubGroupInfo(context_impl &CtxImpl, device_impl &DevImpl,
                                   DeviceKernelInfo &KernelInfo,
                                   ur_kernel_sub_group_info_t InfoCode,
                                   size_t ResultSize, void *Result) {
  FastKernelCacheValPtr CacheVal =
      fetchCachedKernel(CtxImpl, DevImpl, KernelInfo);
  adapter_impl &Adapter = CtxImpl.getAdapter();
  Adapter.call<UrApiKind::urKernelGetSubGroupInfo>(
      CacheVal->MKernelHandle, DevImpl.getHandleRef(), InfoCode, ResultSize,
      Result, nullptr);
}

void queryCachedKernelUrKernelInfo(context_impl &CtxImpl, device_impl &DevImpl,
                                   DeviceKernelInfo &KernelInfo,
                                   ur_kernel_info_t InfoCode, size_t ResultSize,
                                   void *Result) {
  FastKernelCacheValPtr CacheVal =
      fetchCachedKernel(CtxImpl, DevImpl, KernelInfo);
  adapter_impl &Adapter = CtxImpl.getAdapter();
  Adapter.call<UrApiKind::urKernelGetInfo>(CacheVal->MKernelHandle, InfoCode,
                                           ResultSize, Result, nullptr);
}

size_t queryCachedKernelSpillMemSize(context_impl &CtxImpl,
                                     device_impl &DevImpl,
                                     DeviceKernelInfo &KernelInfo) {
  FastKernelCacheValPtr CacheVal =
      fetchCachedKernel(CtxImpl, DevImpl, KernelInfo);
  adapter_impl &Adapter = CtxImpl.getAdapter();
  ur_kernel_handle_t Kernel = CacheVal->MKernelHandle;

  size_t ResultSize = 0;
  Adapter.call<UrApiKind::urKernelGetInfo>(
      Kernel, UR_KERNEL_INFO_SPILL_MEM_SIZE, 0u, nullptr, &ResultSize);
  size_t DeviceCount = ResultSize / sizeof(uint32_t);
  std::vector<uint32_t> Device2SpillMap(DeviceCount);
  Adapter.call<UrApiKind::urKernelGetInfo>(
      Kernel, UR_KERNEL_INFO_SPILL_MEM_SIZE, ResultSize, Device2SpillMap.data(),
      nullptr);

  ur_program_handle_t Program;
  Adapter.call<UrApiKind::urKernelGetInfo>(Kernel, UR_KERNEL_INFO_PROGRAM,
                                           sizeof(ur_program_handle_t),
                                           &Program, nullptr);

  size_t URDevicesSize = 0;
  Adapter.call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                            0u, nullptr, &URDevicesSize);
  std::vector<ur_device_handle_t> URDevices(URDevicesSize /
                                            sizeof(ur_device_handle_t));
  Adapter.call<UrApiKind::urProgramGetInfo>(Program, UR_PROGRAM_INFO_DEVICES,
                                            URDevicesSize, URDevices.data(),
                                            nullptr);
  assert(Device2SpillMap.size() == URDevices.size());

  ur_device_handle_t DevHandle = DevImpl.getHandleRef();
  for (size_t Idx = 0; Idx < URDevices.size(); ++Idx) {
    if (URDevices[Idx] == DevHandle)
      return size_t{Device2SpillMap[Idx]};
  }
  throw exception(
      make_error_code(errc::runtime),
      "ext::intel::info::kernel::spill_memory_size failed to retrieve "
      "the requested value");
}

} // namespace detail
} // namespace _V1
} // namespace sycl
