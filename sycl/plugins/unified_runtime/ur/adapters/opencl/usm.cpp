//===--------- usm.cpp - OpenCL Adapter -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t, size_t size, void **ppMem) {

  void *Ptr = nullptr;
  ur_result_t RetVal = UR_RESULT_ERROR_INVALID_OPERATION;
  uint32_t Alignment = pUSMDesc ? pUSMDesc->align : 0;

  cl_mem_alloc_flags_intel Flags = 0;
  cl_mem_properties_intel Properties[3];

  if (pUSMDesc && pUSMDesc->pNext &&
      static_cast<const ur_base_desc_t *>(pUSMDesc->pNext)->stype ==
          UR_STRUCTURE_TYPE_USM_HOST_DESC) {
    const auto *HostDesc =
        static_cast<const ur_usm_host_desc_t *>(pUSMDesc->pNext);

    if (HostDesc->flags & UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT) {
      Flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL;
    }
    Properties[0] = CL_MEM_ALLOC_FLAGS_INTEL;
    Properties[1] = Flags;
    Properties[2] = 0;
  } else {
    Properties[0] = 0;
  }

  // First we need to look up the function pointer
  clHostMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  RetVal = cl_ext::getExtFuncFromContext<clHostMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clHostMemAllocINTELCache,
      cl_ext::HostMemAllocName, &FuncPtr);

  if (FuncPtr) {
    Ptr = FuncPtr(CLContext, Properties, size, Alignment,
                  cl_adapter::cast<cl_int *>(&RetVal));
  }

  *ppMem = Ptr;

  // ensure we aligned the allocation correctly
  if (RetVal == UR_RESULT_SUCCESS && Alignment != 0)
    assert(reinterpret_cast<std::uintptr_t>(*ppMem) % Alignment == 0 &&
           "allocation not aligned correctly");

  return RetVal;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t,
                 size_t size, void **ppMem) {

  void *Ptr = nullptr;
  ur_result_t RetVal = UR_RESULT_ERROR_INVALID_OPERATION;
  uint32_t Alignment = pUSMDesc ? pUSMDesc->align : 0;

  cl_mem_alloc_flags_intel Flags = 0;
  cl_mem_properties_intel Properties[3];
  if (pUSMDesc && pUSMDesc->pNext &&
      static_cast<const ur_base_desc_t *>(pUSMDesc->pNext)->stype ==
          UR_STRUCTURE_TYPE_USM_DEVICE_DESC) {
    const auto *HostDesc =
        static_cast<const ur_usm_device_desc_t *>(pUSMDesc->pNext);

    if (HostDesc->flags & UR_USM_DEVICE_MEM_FLAG_INITIAL_PLACEMENT) {
      Flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL;
    }
    if (HostDesc->flags & UR_USM_DEVICE_MEM_FLAG_WRITE_COMBINED) {
      Flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL;
    }
    Properties[0] = CL_MEM_ALLOC_FLAGS_INTEL;
    Properties[1] = Flags;
    Properties[2] = 0;
  } else {
    Properties[0] = 0;
  }

  // First we need to look up the function pointer
  clDeviceMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  RetVal = cl_ext::getExtFuncFromContext<clDeviceMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clDeviceMemAllocINTELCache,
      cl_ext::DeviceMemAllocName, &FuncPtr);

  if (FuncPtr) {
    Ptr = FuncPtr(CLContext, cl_adapter::cast<cl_device_id>(hDevice),
                  cl_adapter::cast<cl_mem_properties_intel *>(Properties), size,
                  Alignment, cl_adapter::cast<cl_int *>(&RetVal));
  }

  *ppMem = Ptr;

  // ensure we aligned the allocation correctly
  if (RetVal == UR_RESULT_SUCCESS && Alignment != 0)
    assert(reinterpret_cast<std::uintptr_t>(*ppMem) % Alignment == 0 &&
           "allocation not aligned correctly");

  return RetVal;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t,
                 size_t size, void **ppMem) {

  void *Ptr = nullptr;
  ur_result_t RetVal = UR_RESULT_ERROR_INVALID_OPERATION;
  uint32_t Alignment = pUSMDesc ? pUSMDesc->align : 0;

  cl_mem_alloc_flags_intel Flags = 0;
  const auto *NextStruct =
      (pUSMDesc ? static_cast<const ur_base_desc_t *>(pUSMDesc->pNext)
                : nullptr);
  while (NextStruct) {
    if (NextStruct->stype == UR_STRUCTURE_TYPE_USM_HOST_DESC) {
      const auto *HostDesc =
          reinterpret_cast<const ur_usm_host_desc_t *>(NextStruct);
      if (HostDesc->flags & UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT) {
        Flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL;
      }
    } else if (NextStruct->stype == UR_STRUCTURE_TYPE_USM_DEVICE_DESC) {
      const auto *DevDesc =
          reinterpret_cast<const ur_usm_device_desc_t *>(NextStruct);
      if (DevDesc->flags & UR_USM_DEVICE_MEM_FLAG_INITIAL_PLACEMENT) {
        Flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL;
      }
      if (DevDesc->flags & UR_USM_DEVICE_MEM_FLAG_WRITE_COMBINED) {
        Flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL;
      }
    }
    NextStruct = static_cast<const ur_base_desc_t *>(NextStruct->pNext);
  }

  cl_mem_properties_intel Properties[3] = {CL_MEM_ALLOC_FLAGS_INTEL, Flags, 0};

  // Passing a flags value of 0 doesn't work, so truncate the properties
  if (Flags == 0) {
    Properties[0] = 0;
  }

  // First we need to look up the function pointer
  clSharedMemAllocINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  RetVal = cl_ext::getExtFuncFromContext<clSharedMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clSharedMemAllocINTELCache,
      cl_ext::SharedMemAllocName, &FuncPtr);

  if (FuncPtr) {
    Ptr = FuncPtr(CLContext, cl_adapter::cast<cl_device_id>(hDevice),
                  cl_adapter::cast<cl_mem_properties_intel *>(Properties), size,
                  Alignment, cl_adapter::cast<cl_int *>(&RetVal));
  }

  *ppMem = Ptr;

  assert(Alignment == 0 ||
         (RetVal == UR_RESULT_SUCCESS &&
          reinterpret_cast<std::uintptr_t>(*ppMem) % Alignment == 0));
  return RetVal;
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

  clEnqueueMemFillINTEL_fn FuncPtr = nullptr;
  ur_result_t RetVal = cl_ext::getExtFuncFromContext<clEnqueueMemFillINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clEnqueueMemFillINTELCache,
      cl_ext::EnqueueMemFillName, &FuncPtr);

  if (FuncPtr) {
    RetVal = mapCLErrorToUR(
        FuncPtr(cl_adapter::cast<cl_command_queue>(hQueue), ptr, pPattern,
                patternSize, size, numEventsInWaitList,
                cl_adapter::cast<const cl_event *>(phEventWaitList),
                cl_adapter::cast<cl_event *>(phEvent)));
  }

  return RetVal;
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

  clEnqueueMemcpyINTEL_fn FuncPtr = nullptr;
  ur_result_t RetVal = cl_ext::getExtFuncFromContext<clEnqueueMemcpyINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clEnqueueMemcpyINTELCache,
      cl_ext::EnqueueMemcpyName, &FuncPtr);

  if (FuncPtr) {
    RetVal = mapCLErrorToUR(
        FuncPtr(cl_adapter::cast<cl_command_queue>(hQueue), blocking, pDst,
                pSrc, size, numEventsInWaitList,
                cl_adapter::cast<const cl_event *>(phEventWaitList),
                cl_adapter::cast<cl_event *>(phEvent)));
  }

  return RetVal;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, [[maybe_unused]] const void *pMem,
    [[maybe_unused]] size_t size, ur_usm_migration_flags_t flags,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  // flags is currently unused so fail if set
  if (flags != 0)
    return UR_RESULT_ERROR_INVALID_VALUE;

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
  return UR_RESULT_ERROR_INVALID_OPERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    [[maybe_unused]] ur_queue_handle_t hQueue, [[maybe_unused]] bool blocking,
    [[maybe_unused]] void *pDst, [[maybe_unused]] size_t dstPitch,
    [[maybe_unused]] const void *pSrc, [[maybe_unused]] size_t srcPitch,
    [[maybe_unused]] size_t width, [[maybe_unused]] size_t height,
    [[maybe_unused]] uint32_t numEventsInWaitList,
    [[maybe_unused]] const ur_event_handle_t *phEventWaitList,
    [[maybe_unused]] ur_event_handle_t *phEvent) {
  return UR_RESULT_ERROR_INVALID_OPERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {

  clGetMemAllocInfoINTEL_fn FuncPtr = nullptr;
  cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
  ur_result_t RetVal = cl_ext::getExtFuncFromContext<clGetMemAllocInfoINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clGetMemAllocInfoINTELCache,
      cl_ext::GetMemAllocInfoName, &FuncPtr);

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
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  if (FuncPtr) {
    RetVal =
        mapCLErrorToUR(FuncPtr(cl_adapter::cast<cl_context>(hContext), pMem,
                               PropNameCL, propSize, pPropValue, pPropSizeRet));
    if (RetVal == UR_RESULT_SUCCESS && pPropValue &&
        propName == UR_USM_ALLOC_INFO_TYPE) {
      auto *AllocTypeCL =
          static_cast<cl_unified_shared_memory_type_intel *>(pPropValue);
      ur_usm_type_t AllocTypeUR;
      switch (*AllocTypeCL) {
      case CL_MEM_TYPE_HOST_INTEL:
        AllocTypeUR = UR_USM_TYPE_HOST;
        break;
      case CL_MEM_TYPE_DEVICE_INTEL:
        AllocTypeUR = UR_USM_TYPE_DEVICE;
        break;
      case CL_MEM_TYPE_SHARED_INTEL:
        AllocTypeUR = UR_USM_TYPE_SHARED;
        break;
      case CL_MEM_TYPE_UNKNOWN_INTEL:
      default:
        AllocTypeUR = UR_USM_TYPE_UNKNOWN;
        break;
      }
      auto *AllocTypeOut = static_cast<ur_usm_type_t *>(pPropValue);
      *AllocTypeOut = AllocTypeUR;
    }
  }

  return RetVal;
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
