//==-------------- usm_dispatch.hpp - SYCL USM Dispatch --------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/sycl/detail/clusm.hpp>

#include <memory>

namespace cl {
namespace sycl {
namespace detail {
namespace usm {

class USMDispatcher {
public:
  USMDispatcher(cl_platform_id Platform,
                const vector_class<RT::PiDevice> &DeviceIds);

  void *hostMemAlloc(pi_context Context, cl_mem_properties_intel *Properties,
                     size_t Size, pi_uint32 Alignment, pi_result *ErrcodeRet);
  void *deviceMemAlloc(pi_context Context, pi_device Device,
                       cl_mem_properties_intel *Properties, size_t Size,
                       pi_uint32 Alignment, pi_result *ErrcodeRet);
  void *sharedMemAlloc(pi_context Context, pi_device Device,
                       cl_mem_properties_intel *Properties, size_t Size,
                       pi_uint32 Alignment, pi_result *ErrcodeRet);
  pi_result memFree(pi_context Context, void *Ptr);
  pi_result setKernelArgMemPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   const void *ArgValue);
  void setKernelIndirectAccess(pi_kernel Kernel, pi_queue Queue);
  pi_result enqueueMemset(pi_queue Queue, void *Ptr, pi_int32 Value,
                          size_t Count, pi_uint32 NumEventsInWaitList,
                          const pi_event *EventWaitList, pi_event *Event);
  pi_result enqueueMemcpy(pi_queue Queue, pi_bool Blocking, void *DestPtr,
                          const void *SrcPtr, size_t Size,
                          pi_uint32 NumEventsInWaitList,
                          const pi_event *EventWaitList, pi_event *Event);
  pi_result enqueueMigrateMem(pi_queue Queue, const void *Ptr, size_t Size,
                              cl_mem_migration_flags Flags,
                              pi_uint32 NumEventsInWaitList,
                              const pi_event *EventWaitList, pi_event *Event);
  pi_result getMemAllocInfo(pi_context Context, const void *Ptr,
                            cl_mem_info_intel ParamName, size_t ParamValueSize,
                            void *ParamValue, size_t *ParamValueSizeRet);
  void memAdvise(pi_queue Queue, const void *Ptr, size_t Length, int Advice,
                 pi_event *Event);
  pi_result enqueuePrefetch(pi_queue Queue, void *Ptr, size_t Size,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *Event);

private:
  bool mEmulated = false;
  bool mSupported = false;
  std::unique_ptr<CLUSM> mEmulator;

  clHostMemAllocINTEL_fn pfn_clHostMemAllocINTEL = nullptr;
  clDeviceMemAllocINTEL_fn pfn_clDeviceMemAllocINTEL = nullptr;
  clSharedMemAllocINTEL_fn pfn_clSharedMemAllocINTEL = nullptr;
  clMemFreeINTEL_fn pfn_clMemFreeINTEL = nullptr;
  clGetMemAllocInfoINTEL_fn pfn_clGetMemAllocInfoINTEL = nullptr;
  clSetKernelArgMemPointerINTEL_fn pfn_clSetKernelArgMemPointerINTEL = nullptr;
  clEnqueueMemsetINTEL_fn pfn_clEnqueueMemsetINTEL = nullptr;
  clEnqueueMemcpyINTEL_fn pfn_clEnqueueMemcpyINTEL = nullptr;
  clEnqueueMigrateMemINTEL_fn pfn_clEnqueueMigrateMemINTEL = nullptr;
  clEnqueueMemAdviseINTEL_fn pfn_clEnqueueMemAdviseINTEL = nullptr;
};

} // namespace usm
} // namespace detail
} // namespace sycl
} // namespace cl
