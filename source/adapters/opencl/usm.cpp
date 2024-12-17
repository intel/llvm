//===--------- usm.cpp - OpenCL Adapter -------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur/ur.hpp>

#include "common.hpp"
#include "usm.hpp"

template <class T>
void AllocDeleterCallback(cl_event event, cl_int, void *pUserData) {
  clReleaseEvent(event);
  auto Info = static_cast<T *>(pUserData);
  delete Info;
}

namespace umf {
ur_result_t getProviderNativeError(const char *, int32_t) {
  // TODO: implement when UMF supports OpenCL
  return UR_RESULT_ERROR_UNKNOWN;
}
} // namespace umf

inline cl_mem_alloc_flags_intel
hostDescToClFlags(const ur_usm_host_desc_t &desc) {
  cl_mem_alloc_flags_intel allocFlags = 0;
  if (desc.flags & UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT) {
    allocFlags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL;
  }
  return allocFlags;
}

inline cl_mem_alloc_flags_intel
deviceDescToClFlags(const ur_usm_device_desc_t &desc) {
  cl_mem_alloc_flags_intel allocFlags = 0;
  if (desc.flags & UR_USM_DEVICE_MEM_FLAG_INITIAL_PLACEMENT) {
    allocFlags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL;
  }
  if (desc.flags & UR_USM_DEVICE_MEM_FLAG_WRITE_COMBINED) {
    allocFlags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL;
  }
  return allocFlags;
}

ur_result_t
usmDescToCLMemProperties(const ur_base_desc_t *Desc,
                         std::vector<cl_mem_properties_intel> &Properties) {
  cl_mem_alloc_flags_intel AllocFlags = 0;
  const auto *Next = Desc;
  do {
    switch (Next->stype) {
    case UR_STRUCTURE_TYPE_USM_HOST_DESC: {
      auto HostDesc = reinterpret_cast<const ur_usm_host_desc_t *>(Next);
      if (UR_USM_HOST_MEM_FLAGS_MASK & HostDesc->flags) {
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
      }
      AllocFlags |= hostDescToClFlags(*HostDesc);
      break;
    }
    case UR_STRUCTURE_TYPE_USM_DEVICE_DESC: {
      auto DeviceDesc = reinterpret_cast<const ur_usm_device_desc_t *>(Next);
      if (UR_USM_HOST_MEM_FLAGS_MASK & DeviceDesc->flags) {
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
      }
      AllocFlags |= deviceDescToClFlags(*DeviceDesc);
      break;
    }
    case UR_STRUCTURE_TYPE_USM_ALLOC_LOCATION_DESC: {
      auto LocationDesc =
          reinterpret_cast<const ur_usm_alloc_location_desc_t *>(Next);
      Properties.push_back(CL_MEM_ALLOC_BUFFER_LOCATION_INTEL);
      // CL bitfields are cl_ulong
      Properties.push_back(static_cast<cl_ulong>(LocationDesc->location));
      break;
    }
    default:
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    Next = static_cast<const ur_base_desc_t *>(Next->pNext);
  } while (Next);

  if (AllocFlags) {
    Properties.push_back(CL_MEM_ALLOC_FLAGS_INTEL);
    Properties.push_back(AllocFlags);
  }
  Properties.push_back(0);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t, size_t size, void **ppMem) {

  void *Ptr = nullptr;
  uint32_t Alignment = pUSMDesc ? pUSMDesc->align : 0;

  if (pUSMDesc && pUSMDesc->align != 0 &&
      ((pUSMDesc->align & (pUSMDesc->align - 1)) != 0)) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  std::vector<cl_mem_properties_intel> AllocProperties;
  if (pUSMDesc && pUSMDesc->pNext) {
    UR_RETURN_ON_FAILURE(usmDescToCLMemProperties(
        static_cast<const ur_base_desc_t *>(pUSMDesc->pNext), AllocProperties));
  }

  // First we need to look up the function pointer
  clHostMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  if (auto UrResult = cl_ext::getExtFuncFromContext<clHostMemAllocINTEL_fn>(
          CLContext, cl_ext::ExtFuncPtrCache->clHostMemAllocINTELCache,
          cl_ext::HostMemAllocName, &FuncPtr)) {
    return UrResult;
  }

  if (FuncPtr) {
    cl_int ClResult = CL_SUCCESS;
    Ptr = FuncPtr(CLContext,
                  AllocProperties.empty() ? nullptr : AllocProperties.data(),
                  size, Alignment, &ClResult);
    if (ClResult == CL_INVALID_BUFFER_SIZE) {
      return UR_RESULT_ERROR_INVALID_USM_SIZE;
    }
    CL_RETURN_ON_FAILURE(ClResult);
  }

  *ppMem = Ptr;

  assert((Alignment == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % Alignment == 0) &&
         "Allocation not aligned correctly!");

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t,
                 size_t size, void **ppMem) {

  void *Ptr = nullptr;
  uint32_t Alignment = pUSMDesc ? pUSMDesc->align : 0;

  if (pUSMDesc && pUSMDesc->align != 0 &&
      ((pUSMDesc->align & (pUSMDesc->align - 1)) != 0)) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  std::vector<cl_mem_properties_intel> AllocProperties;
  if (pUSMDesc && pUSMDesc->pNext) {
    UR_RETURN_ON_FAILURE(usmDescToCLMemProperties(
        static_cast<const ur_base_desc_t *>(pUSMDesc->pNext), AllocProperties));
  }

  // First we need to look up the function pointer
  clDeviceMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  if (auto UrResult = cl_ext::getExtFuncFromContext<clDeviceMemAllocINTEL_fn>(
          CLContext, cl_ext::ExtFuncPtrCache->clDeviceMemAllocINTELCache,
          cl_ext::DeviceMemAllocName, &FuncPtr)) {
    return UrResult;
  }

  if (FuncPtr) {
    cl_int ClResult = CL_SUCCESS;
    Ptr = FuncPtr(CLContext, cl_adapter::cast<cl_device_id>(hDevice),
                  AllocProperties.empty() ? nullptr : AllocProperties.data(),
                  size, Alignment, &ClResult);
    if (ClResult == CL_INVALID_BUFFER_SIZE) {
      return UR_RESULT_ERROR_INVALID_USM_SIZE;
    }
    CL_RETURN_ON_FAILURE(ClResult);
  }

  *ppMem = Ptr;

  assert((Alignment == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % Alignment == 0) &&
         "Allocation not aligned correctly!");

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t,
                 size_t size, void **ppMem) {

  void *Ptr = nullptr;
  uint32_t Alignment = pUSMDesc ? pUSMDesc->align : 0;

  if (pUSMDesc && pUSMDesc->align != 0 &&
      ((pUSMDesc->align & (pUSMDesc->align - 1)) != 0)) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  std::vector<cl_mem_properties_intel> AllocProperties;
  if (pUSMDesc && pUSMDesc->pNext) {
    UR_RETURN_ON_FAILURE(usmDescToCLMemProperties(
        static_cast<const ur_base_desc_t *>(pUSMDesc->pNext), AllocProperties));
  }

  // First we need to look up the function pointer
  clSharedMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  if (auto UrResult = cl_ext::getExtFuncFromContext<clSharedMemAllocINTEL_fn>(
          CLContext, cl_ext::ExtFuncPtrCache->clSharedMemAllocINTELCache,
          cl_ext::SharedMemAllocName, &FuncPtr)) {
    return UrResult;
  }

  if (FuncPtr) {
    cl_int ClResult = CL_SUCCESS;
    Ptr = FuncPtr(CLContext, cl_adapter::cast<cl_device_id>(hDevice),
                  AllocProperties.empty() ? nullptr : AllocProperties.data(),
                  size, Alignment, cl_adapter::cast<cl_int *>(&ClResult));
    if (ClResult == CL_INVALID_BUFFER_SIZE) {
      return UR_RESULT_ERROR_INVALID_USM_SIZE;
    }
    CL_RETURN_ON_FAILURE(ClResult);
  }

  *ppMem = Ptr;

  assert((Alignment == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % Alignment == 0) &&
         "Allocation not aligned correctly!");
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {

  // Use a blocking free to avoid issues with indirect access from kernels that
  // might be still running.
  clMemBlockingFreeINTEL_fn FuncPtr = nullptr;

  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  ur_result_t RetVal = UR_RESULT_ERROR_INVALID_OPERATION;
  RetVal = cl_ext::getExtFuncFromContext<clMemBlockingFreeINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clMemBlockingFreeINTELCache,
      cl_ext::MemBlockingFreeName, &FuncPtr);

  if (FuncPtr) {
    RetVal = mapCLErrorToUR(FuncPtr(CLContext, pMem));
  }

  return RetVal;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr = clGetCommandQueueInfo(
      cl_adapter::cast<cl_command_queue>(hQueue), CL_QUEUE_CONTEXT,
      sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return mapCLErrorToUR(CLErr);
  }

  if (patternSize <= 128 && isPowerOf2(patternSize)) {
    clEnqueueMemFillINTEL_fn EnqueueMemFill = nullptr;
    UR_RETURN_ON_FAILURE(
        cl_ext::getExtFuncFromContext<clEnqueueMemFillINTEL_fn>(
            CLContext, cl_ext::ExtFuncPtrCache->clEnqueueMemFillINTELCache,
            cl_ext::EnqueueMemFillName, &EnqueueMemFill));

    CL_RETURN_ON_FAILURE(
        EnqueueMemFill(cl_adapter::cast<cl_command_queue>(hQueue), ptr,
                       pPattern, patternSize, size, numEventsInWaitList,
                       cl_adapter::cast<const cl_event *>(phEventWaitList),
                       cl_adapter::cast<cl_event *>(phEvent)));
    return UR_RESULT_SUCCESS;
  }

  // OpenCL only supports pattern sizes which are powers of 2 and are as large
  // as the largest CL type (double16/long16 - 128 bytes), anything larger or
  // not a power of 2, we need to do on the host side and copy it into the
  // target allocation.
  clHostMemAllocINTEL_fn HostMemAlloc = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clHostMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clHostMemAllocINTELCache,
      cl_ext::HostMemAllocName, &HostMemAlloc));

  clEnqueueMemcpyINTEL_fn USMMemcpy = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clEnqueueMemcpyINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clEnqueueMemcpyINTELCache,
      cl_ext::EnqueueMemcpyName, &USMMemcpy));

  clMemBlockingFreeINTEL_fn USMFree = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clMemBlockingFreeINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clMemBlockingFreeINTELCache,
      cl_ext::MemBlockingFreeName, &USMFree));

  cl_int ClErr = CL_SUCCESS;
  auto HostBuffer =
      static_cast<uint8_t *>(HostMemAlloc(CLContext, nullptr, size, 0, &ClErr));
  CL_RETURN_ON_FAILURE(ClErr);

  auto *End = HostBuffer + size;
  for (auto *Iter = HostBuffer; Iter < End; Iter += patternSize) {
    std::memcpy(Iter, pPattern, patternSize);
  }

  cl_event CopyEvent = nullptr;
  CL_RETURN_ON_FAILURE(USMMemcpy(
      cl_adapter::cast<cl_command_queue>(hQueue), false, ptr, HostBuffer, size,
      numEventsInWaitList, cl_adapter::cast<const cl_event *>(phEventWaitList),
      &CopyEvent));

  if (phEvent) {
    // Since we're releasing this in the callback above we need to retain it
    // here to keep the user copy alive.
    CL_RETURN_ON_FAILURE(clRetainEvent(CopyEvent));
    *phEvent = cl_adapter::cast<ur_event_handle_t>(CopyEvent);
  }

  // This self destructs taking the event and allocation with it.
  auto Info = new AllocDeleterCallbackInfo(USMFree, CLContext, HostBuffer);

  ClErr =
      clSetEventCallback(CopyEvent, CL_COMPLETE,
                         AllocDeleterCallback<AllocDeleterCallbackInfo>, Info);
  if (ClErr != CL_SUCCESS) {
    // We can attempt to recover gracefully by attempting to wait for the copy
    // to finish and deleting the info struct here.
    clWaitForEvents(1, &CopyEvent);
    delete Info;
    clReleaseEvent(CopyEvent);
    CL_RETURN_ON_FAILURE(ClErr);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, const void *pSrc,
    size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr = clGetCommandQueueInfo(
      cl_adapter::cast<cl_command_queue>(hQueue), CL_QUEUE_CONTEXT,
      sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return mapCLErrorToUR(CLErr);
  }

  clGetMemAllocInfoINTEL_fn GetMemAllocInfo = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clGetMemAllocInfoINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clGetMemAllocInfoINTELCache,
      cl_ext::GetMemAllocInfoName, &GetMemAllocInfo));

  clEnqueueMemcpyINTEL_fn USMMemcpy = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clEnqueueMemcpyINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clEnqueueMemcpyINTELCache,
      cl_ext::EnqueueMemcpyName, &USMMemcpy));

  clMemBlockingFreeINTEL_fn USMFree = nullptr;
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clMemBlockingFreeINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clMemBlockingFreeINTELCache,
      cl_ext::MemBlockingFreeName, &USMFree));

  // Check if the two allocations are DEVICE allocations from different
  // devices, if they are we need to do the copy indirectly via a host
  // allocation.
  cl_device_id SrcDevice = 0, DstDevice = 0;
  CL_RETURN_ON_FAILURE(
      GetMemAllocInfo(CLContext, pSrc, CL_MEM_ALLOC_DEVICE_INTEL,
                      sizeof(cl_device_id), &SrcDevice, nullptr));
  CL_RETURN_ON_FAILURE(
      GetMemAllocInfo(CLContext, pDst, CL_MEM_ALLOC_DEVICE_INTEL,
                      sizeof(cl_device_id), &DstDevice, nullptr));

  if ((SrcDevice && DstDevice) && SrcDevice != DstDevice) {
    // We need a queue associated with each device, so first figure out which
    // one we weren't given.
    cl_device_id QueueDevice = nullptr;
    CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
        cl_adapter::cast<cl_command_queue>(hQueue), CL_QUEUE_DEVICE,
        sizeof(QueueDevice), &QueueDevice, nullptr));

    cl_command_queue MissingQueue = nullptr, SrcQueue = nullptr,
                     DstQueue = nullptr;
    if (QueueDevice == SrcDevice) {
      MissingQueue = clCreateCommandQueue(CLContext, DstDevice, 0, &CLErr);
      SrcQueue = cl_adapter::cast<cl_command_queue>(hQueue);
      DstQueue = MissingQueue;
    } else {
      MissingQueue = clCreateCommandQueue(CLContext, SrcDevice, 0, &CLErr);
      DstQueue = cl_adapter::cast<cl_command_queue>(hQueue);
      SrcQueue = MissingQueue;
    }
    CL_RETURN_ON_FAILURE(CLErr);

    cl_event HostCopyEvent = nullptr, FinalCopyEvent = nullptr;
    clHostMemAllocINTEL_fn HostMemAlloc = nullptr;
    UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clHostMemAllocINTEL_fn>(
        CLContext, cl_ext::ExtFuncPtrCache->clHostMemAllocINTELCache,
        cl_ext::HostMemAllocName, &HostMemAlloc));

    auto HostAlloc = HostMemAlloc(CLContext, nullptr, size, 0, &CLErr);
    CL_RETURN_ON_FAILURE(CLErr);

    // Now that we've successfully allocated we should try to clean it up if we
    // hit an error somewhere.
    auto checkCLErr = [&](cl_int CLErr) -> ur_result_t {
      if (CLErr != CL_SUCCESS) {
        if (HostCopyEvent) {
          clReleaseEvent(HostCopyEvent);
        }
        if (FinalCopyEvent) {
          clReleaseEvent(FinalCopyEvent);
        }
        USMFree(CLContext, HostAlloc);
        CL_RETURN_ON_FAILURE(CLErr);
      }
      return UR_RESULT_SUCCESS;
    };

    UR_RETURN_ON_FAILURE(checkCLErr(USMMemcpy(
        SrcQueue, blocking, HostAlloc, pSrc, size, numEventsInWaitList,
        cl_adapter::cast<const cl_event *>(phEventWaitList), &HostCopyEvent)));

    UR_RETURN_ON_FAILURE(
        checkCLErr(USMMemcpy(DstQueue, blocking, pDst, HostAlloc, size, 1,
                             &HostCopyEvent, &FinalCopyEvent)));

    // If this is a blocking operation we can do our cleanup immediately,
    // otherwise we need to defer it to an event callback.
    if (blocking) {
      CL_RETURN_ON_FAILURE(USMFree(CLContext, HostAlloc));
      CL_RETURN_ON_FAILURE(clReleaseEvent(HostCopyEvent));
      CL_RETURN_ON_FAILURE(clReleaseCommandQueue(MissingQueue));
      if (phEvent) {
        *phEvent = cl_adapter::cast<ur_event_handle_t>(FinalCopyEvent);
      } else {
        CL_RETURN_ON_FAILURE(clReleaseEvent(FinalCopyEvent));
      }
    } else {
      if (phEvent) {
        *phEvent = cl_adapter::cast<ur_event_handle_t>(FinalCopyEvent);
        // We are going to release this event in our callback so we need to
        // retain if the user wants a copy.
        CL_RETURN_ON_FAILURE(clRetainEvent(FinalCopyEvent));
      }

      // This self destructs taking the event and allocation with it.
      auto DeleterInfo = new AllocDeleterCallbackInfoWithQueue(
          USMFree, CLContext, HostAlloc, MissingQueue);

      CLErr = clSetEventCallback(
          HostCopyEvent, CL_COMPLETE,
          AllocDeleterCallback<AllocDeleterCallbackInfoWithQueue>, DeleterInfo);

      if (CLErr != CL_SUCCESS) {
        // We can attempt to recover gracefully by attempting to wait for the
        // copy to finish and deleting the info struct here.
        clWaitForEvents(1, &HostCopyEvent);
        delete DeleterInfo;
        clReleaseEvent(HostCopyEvent);
        CL_RETURN_ON_FAILURE(CLErr);
      }
    }
  } else {
    CL_RETURN_ON_FAILURE(
        USMMemcpy(cl_adapter::cast<cl_command_queue>(hQueue), blocking, pDst,
                  pSrc, size, numEventsInWaitList,
                  cl_adapter::cast<const cl_event *>(phEventWaitList),
                  cl_adapter::cast<cl_event *>(phEvent)));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, [[maybe_unused]] const void *pMem,
    [[maybe_unused]] size_t size,
    [[maybe_unused]] ur_usm_migration_flags_t flags,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  return mapCLErrorToUR(clEnqueueMarkerWithWaitList(
      cl_adapter::cast<cl_command_queue>(hQueue), numEventsInWaitList,
      cl_adapter::cast<const cl_event *>(phEventWaitList),
      cl_adapter::cast<cl_event *>(phEvent)));

  /*
  // Use this once impls support it.
  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr =
  clGetCommandQueueInfo(cl_adapter::cast<cl_command_queue>(hQueue),
                                       CL_QUEUE_CONTEXT, sizeof(cl_context),
                                       &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return map_cl_error_to_ur(CLErr);
  }

  clEnqueueMigrateMemINTEL_fn FuncPtr;
  ur_result_t Err = cl_ext::getExtFuncFromContext<clEnqueueMigrateMemINTEL_fn>(
      CLContext, "clEnqueueMigrateMemINTEL", &FuncPtr);

  ur_result_t RetVal;
  if (Err != UR_RESULT_SUCCESS) {
    RetVal = Err;
  } else {
    RetVal = map_cl_error_to_ur(
        FuncPtr(cl_adapter::cast<cl_command_queue>(hQueue), pMem, size, flags,
                numEventsInWaitList,
                reinterpret_cast<const cl_event *>(phEventWaitList),
                reinterpret_cast<cl_event *>(phEvent)));
  }
  */
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMAdvise(
    ur_queue_handle_t hQueue, [[maybe_unused]] const void *pMem,
    [[maybe_unused]] size_t size, [[maybe_unused]] ur_usm_advice_flags_t advice,
    ur_event_handle_t *phEvent) {

  return mapCLErrorToUR(clEnqueueMarkerWithWaitList(
      cl_adapter::cast<cl_command_queue>(hQueue), 0, nullptr,
      reinterpret_cast<cl_event *>(phEvent)));

  /*
  // Change to use this once drivers support it.
  // Have to look up the context from the kernel
  cl_context CLContext;
  cl_int CLErr =
  clGetCommandQueueInfo(cl_adapter::cast<cl_command_queue>(hQueue),
                                 CL_QUEUE_CONTEXT,
                                 sizeof(cl_context),
                                 &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return map_cl_error_to_ur(CLErr);
  }

  clEnqueueMemAdviseINTEL_fn FuncPtr;
  ur_result_t Err =
    cl_ext::getExtFuncFromContext<clEnqueueMemAdviseINTEL_fn>(
      CLContext, "clEnqueueMemAdviseINTEL", &FuncPtr);

  ur_result_t RetVal;
  if (Err != UR_RESULT_SUCCESS) {
    RetVal = Err;
  } else {
    RetVal =
  map_cl_error_to_ur(FuncPtr(cl_adapter::cast<cl_command_queue>(hQueue), pMem,
  size, advice, 0, nullptr, reinterpret_cast<cl_event *>(phEvent)));
  }
  */
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    [[maybe_unused]] ur_queue_handle_t hQueue, [[maybe_unused]] void *pMem,
    [[maybe_unused]] size_t pitch, [[maybe_unused]] size_t patternSize,
    [[maybe_unused]] const void *pPattern, [[maybe_unused]] size_t width,
    [[maybe_unused]] size_t height,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, size_t dstPitch,
    const void *pSrc, size_t srcPitch, size_t width, size_t height,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  cl_context CLContext;
  CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(
      cl_adapter::cast<cl_command_queue>(hQueue), CL_QUEUE_CONTEXT,
      sizeof(cl_context), &CLContext, nullptr));

  clEnqueueMemcpyINTEL_fn FuncPtr = nullptr;
  ur_result_t RetVal = cl_ext::getExtFuncFromContext<clEnqueueMemcpyINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clEnqueueMemcpyINTELCache,
      cl_ext::EnqueueMemcpyName, &FuncPtr);

  if (!FuncPtr) {
    return RetVal;
  }

  std::vector<cl_event> Events(height);
  for (size_t HeightIndex = 0; HeightIndex < height; HeightIndex++) {
    cl_event Event = nullptr;
    auto ClResult =
        FuncPtr(cl_adapter::cast<cl_command_queue>(hQueue), false,
                static_cast<uint8_t *>(pDst) + dstPitch * HeightIndex,
                static_cast<const uint8_t *>(pSrc) + srcPitch * HeightIndex,
                width, numEventsInWaitList,
                cl_adapter::cast<const cl_event *>(phEventWaitList), &Event);
    Events[HeightIndex] = Event;
    if (ClResult != CL_SUCCESS) {
      for (const auto &E : Events) {
        clReleaseEvent(E);
      }
      CL_RETURN_ON_FAILURE(ClResult);
    }
  }
  cl_int ClResult = CL_SUCCESS;
  if (blocking) {
    ClResult = clWaitForEvents(Events.size(), Events.data());
  }
  if (phEvent && ClResult == CL_SUCCESS) {
    ClResult = clEnqueueBarrierWithWaitList(
        cl_adapter::cast<cl_command_queue>(hQueue), Events.size(),
        Events.data(), cl_adapter::cast<cl_event *>(phEvent));
  }
  for (const auto &E : Events) {
    CL_RETURN_ON_FAILURE(clReleaseEvent(E));
  }
  CL_RETURN_ON_FAILURE(ClResult)
  return UR_RESULT_SUCCESS;
}

ur_usm_type_t
mapCLUSMTypeToUR(const cl_unified_shared_memory_type_intel &Type) {
  switch (Type) {
  case CL_MEM_TYPE_HOST_INTEL:
    return UR_USM_TYPE_HOST;
  case CL_MEM_TYPE_DEVICE_INTEL:
    return UR_USM_TYPE_DEVICE;
  case CL_MEM_TYPE_SHARED_INTEL:
    return UR_USM_TYPE_SHARED;
  case CL_MEM_TYPE_UNKNOWN_INTEL:
  default:
    return UR_USM_TYPE_UNKNOWN;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {

  clGetMemAllocInfoINTEL_fn GetMemAllocInfo = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clGetMemAllocInfoINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clGetMemAllocInfoINTELCache,
      cl_ext::GetMemAllocInfoName, &GetMemAllocInfo));

  cl_mem_info_intel PropNameCL;
  switch (propName) {
  case UR_USM_ALLOC_INFO_TYPE:
    PropNameCL = CL_MEM_ALLOC_TYPE_INTEL;
    break;
  case UR_USM_ALLOC_INFO_BASE_PTR:
    PropNameCL = CL_MEM_ALLOC_BASE_PTR_INTEL;
    break;
  case UR_USM_ALLOC_INFO_SIZE:
    PropNameCL = CL_MEM_ALLOC_SIZE_INTEL;
    break;
  case UR_USM_ALLOC_INFO_DEVICE:
    PropNameCL = CL_MEM_ALLOC_DEVICE_INTEL;
    break;
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  size_t CheckPropSize = 0;
  cl_int ClErr =
      GetMemAllocInfo(cl_adapter::cast<cl_context>(hContext), pMem, PropNameCL,
                      propSize, pPropValue, &CheckPropSize);
  if (pPropValue && CheckPropSize != propSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }
  CL_RETURN_ON_FAILURE(ClErr);
  if (pPropSizeRet) {
    *pPropSizeRet = CheckPropSize;
  }

  if (pPropValue && propName == UR_USM_ALLOC_INFO_TYPE) {
    *static_cast<ur_usm_type_t *>(pPropValue) = mapCLUSMTypeToUR(
        *static_cast<cl_unified_shared_memory_type_intel *>(pPropValue));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMImportExp([[maybe_unused]] ur_context_handle_t Context,
               [[maybe_unused]] void *HostPtr, [[maybe_unused]] size_t Size) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMReleaseExp([[maybe_unused]] ur_context_handle_t Context,
                [[maybe_unused]] void *HostPtr) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolCreate([[maybe_unused]] ur_context_handle_t hContext,
                [[maybe_unused]] ur_usm_pool_desc_t *pPoolDesc,
                [[maybe_unused]] ur_usm_pool_handle_t *ppPool) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRetain([[maybe_unused]] ur_usm_pool_handle_t pPool) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRelease([[maybe_unused]] ur_usm_pool_handle_t pPool) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetInfo(
    [[maybe_unused]] ur_usm_pool_handle_t hPool,
    [[maybe_unused]] ur_usm_pool_info_t propName,
    [[maybe_unused]] size_t propSize, [[maybe_unused]] void *pPropValue,
    [[maybe_unused]] size_t *pPropSizeRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
