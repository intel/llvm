//===----------- memory.cpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <unordered_set>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "adapter.hpp"
#include "context.hpp"
#include "device.hpp"
#include "memory.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {

  // TODO: We can avoid the initial copy with USE_HOST_POINTER by implementing
  // something like olMemRegister
  const bool PerformInitialCopy =
      (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) ||
      (flags & UR_MEM_FLAG_USE_HOST_POINTER);

  void *Ptr = nullptr;
  auto HostPtr = pProperties ? pProperties->pHost : nullptr;
  auto OffloadDevice = hContext->Device->OffloadDevice;
  auto AllocMode = BufferMem::AllocMode::Default;

  if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    OL_RETURN_ON_ERR(
        olMemAlloc(OffloadDevice, OL_ALLOC_TYPE_HOST, size, &HostPtr));

    // TODO: We (probably) need something like cuMemHostGetDevicePointer
    // for this to work everywhere. For now assume the managed host pointer is
    // device-accessible.
    Ptr = HostPtr;
    AllocMode = BufferMem::AllocMode::AllocHostPtr;
  } else {
    OL_RETURN_ON_ERR(
        olMemAlloc(OffloadDevice, OL_ALLOC_TYPE_DEVICE, size, &Ptr));
    if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
      AllocMode = BufferMem::AllocMode::CopyIn;
    }
  }

  ur_mem_handle_t ParentBuffer = nullptr;
  auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
      hContext, ParentBuffer, flags, AllocMode, Ptr, HostPtr, size});

  if (PerformInitialCopy) {
    OL_RETURN_ON_ERR(olMemcpy(nullptr, Ptr, OffloadDevice, HostPtr,
                              Adapter->HostDevice, size));
  }

  *phBuffer = URMemObj.release();

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  hMem->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  if (--hMem->RefCount > 0) {
    return UR_RESULT_SUCCESS;
  }

  std::unique_ptr<ur_mem_handle_t_> MemObjPtr(hMem);
  if (auto *BufferImpl = MemObjPtr->AsBufferMem()) {
    // Subbuffers should not free their parents
    if (!BufferImpl->Parent) {
      // TODO: Handle registered host memory
      OL_RETURN_ON_ERR(olMemFree(BufferImpl->Ptr));
    } else {
      return urMemRelease(BufferImpl->Parent);
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t MemInfoType,
                                                 size_t propSize,
                                                 void *pMemInfo,
                                                 size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pMemInfo, pPropSizeRet);

  switch (MemInfoType) {
  case UR_MEM_INFO_SIZE: {
    return ReturnValue(std::get<BufferMem>(hMemory->Mem).Size);
  }
  case UR_MEM_INFO_CONTEXT: {
    return ReturnValue(hMemory->getContext());
  }
  case UR_MEM_INFO_REFERENCE_COUNT: {
    return ReturnValue(hMemory->RefCount.load());
  }

  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t /*BufferCreateType*/,
    const ur_buffer_region_t *pRegion, ur_mem_handle_t *phMem) {
  auto *SrcBuffer = hBuffer->AsBufferMem();
  if (!SrcBuffer || SrcBuffer->Parent) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  // Default value for flags means UR_MEM_FLAG_READ_WRITE.
  if (flags == 0) {
    flags = UR_MEM_FLAG_READ_WRITE;
  }
  UR_ASSERT(subBufferFlagsAreLegal(hBuffer->MemFlags, flags),
            UR_RESULT_ERROR_INVALID_VALUE);

  UR_ASSERT(((pRegion->origin + pRegion->size) <= SrcBuffer->getSize()),
            UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  void *DeviceBase =
      reinterpret_cast<uint8_t *>(SrcBuffer->Ptr) + pRegion->origin;
  void *HostBase =
      reinterpret_cast<uint8_t *>(SrcBuffer->HostPtr) + pRegion->origin;
  auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
      hBuffer->getContext(), hBuffer, flags, SrcBuffer->MemAllocMode,
      DeviceBase, HostBase, pRegion->size});
  *phMem = URMemObj.release();

  return urMemRetain(hBuffer);
}

UR_APIEXPORT ur_result_t UR_APICALL
urMemImageCreate(ur_context_handle_t, ur_mem_flags_t, const ur_image_format_t *,
                 const ur_image_desc_t *, void *, ur_mem_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t, const ur_image_format_t *,
    const ur_image_desc_t *, const ur_mem_native_properties_t *,
    ur_mem_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(ur_mem_handle_t,
                                                      ur_image_info_t, size_t,
                                                      void *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCGetMemHandleExp(ur_context_handle_t,
                                                         void *, void *,
                                                         size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCPutMemHandleExp(ur_context_handle_t,
                                                         void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCOpenMemHandleExp(ur_context_handle_t,
                                                          ur_device_handle_t,
                                                          void *, size_t,
                                                          void **) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCCloseMemHandleExp(ur_context_handle_t,
                                                           void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
