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
    auto Res = olMemAlloc(OffloadDevice, OL_ALLOC_TYPE_HOST, size, &HostPtr);
    if (Res) {
      return offloadResultToUR(Res);
    }
    // TODO: We (probably) need something like cuMemHostGetDevicePointer
    // for this to work everywhere. For now assume the managed host pointer is
    // device-accessible.
    Ptr = HostPtr;
    AllocMode = BufferMem::AllocMode::AllocHostPtr;
  } else {
    auto Res = olMemAlloc(OffloadDevice, OL_ALLOC_TYPE_DEVICE, size, &Ptr);
    if (Res) {
      return offloadResultToUR(Res);
    }
    if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
      AllocMode = BufferMem::AllocMode::CopyIn;
    }
  }

  ur_mem_handle_t ParentBuffer = nullptr;
  auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
      hContext, ParentBuffer, flags, AllocMode, Ptr, HostPtr, size});

  if (PerformInitialCopy) {
    auto Res = olMemcpy(nullptr, Ptr, OffloadDevice, HostPtr,
                        Adapter->HostDevice, size, nullptr);
    if (Res) {
      return offloadResultToUR(Res);
    }
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
  if (hMem->MemType == ur_mem_handle_t_::Type::Buffer) {
    // TODO: Handle registered host memory
    auto &BufferImpl = std::get<BufferMem>(MemObjPtr->Mem);
    auto Res = olMemFree(BufferImpl.Ptr);
    if (Res) {
      return offloadResultToUR(Res);
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
