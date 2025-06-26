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

void *BufferMem::getPtr(ur_device_handle_t Device) const noexcept {
  // Create the allocation for this device if needed
  OuterMemStruct->prepareDeviceAllocation(Device);
  return Ptrs[OuterMemStruct->Context->getDeviceIndex(Device)];
}

ur_result_t enqueueMigrateBufferToDevice(ur_mem_handle_t Mem,
                                         ur_device_handle_t Device,
                                         ol_queue_handle_t Queue) {
  auto &Buffer = std::get<BufferMem>(Mem->Mem);
  if (Mem->LastQueueWritingToMemObj == nullptr) {
    // Device allocation being initialized from host for the first time
    if (Buffer.HostPtr) {
      OL_RETURN_ON_ERR(olMemcpy(Queue, Buffer.getPtr(Device),
                                Device->OffloadDevice, Buffer.HostPtr,
                                Adapter->HostDevice, Buffer.Size, nullptr));
    }
  } else if (Mem->LastQueueWritingToMemObj->Device != Device) {
    auto LastDevice = Mem->LastQueueWritingToMemObj->Device;
    OL_RETURN_ON_ERR(olMemcpy(Queue, Buffer.getPtr(Device),
                              Device->OffloadDevice, Buffer.getPtr(LastDevice),
                              LastDevice->OffloadDevice, Buffer.Size, nullptr));
  }
  return UR_RESULT_SUCCESS;
}

// TODO: Check lock in cuda adapter
ur_result_t ur_mem_handle_t_::enqueueMigrateMemoryToDeviceIfNeeded(
    const ur_device_handle_t Device, ol_queue_handle_t Queue) {
  UR_ASSERT(Device, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  // Device allocation has already been initialized with most up to date
  // data in buffer
  if (DeviceIsUpToDate[getContext()->getDeviceIndex(Device)]) {
    return UR_RESULT_SUCCESS;
  }

  return enqueueMigrateBufferToDevice(this, Device, Queue);
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {

  // TODO: We can avoid the initial copy with USE_HOST_POINTER by implementing
  // something like olMemRegister
  const bool PerformInitialCopy =
      (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) ||
      (flags & UR_MEM_FLAG_USE_HOST_POINTER);

  auto HostPtr = pProperties ? pProperties->pHost : nullptr;
  auto AllocMode = BufferMem::AllocMode::Default;

  if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    // Allocate on the first device, which will be valid on all devices in the
    // context
    OL_RETURN_ON_ERR(olMemAlloc(hContext->Devices[0]->OffloadDevice,
                                OL_ALLOC_TYPE_HOST, size, &HostPtr));

    // TODO: We (probably) need something like cuMemHostGetDevicePointer
    // for this to work everywhere. For now assume the managed host pointer is
    // device-accessible.
    AllocMode = BufferMem::AllocMode::AllocHostPtr;
  } else {
    if (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
      AllocMode = BufferMem::AllocMode::CopyIn;
    }
  }

  ur_mem_handle_t ParentBuffer = nullptr;
  auto URMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
      hContext, ParentBuffer, flags, AllocMode, HostPtr, size});

  if (PerformInitialCopy && HostPtr) {
    // Copy per device
    for (auto Device : hContext->Devices) {
      const auto &Ptr = std::get<BufferMem>(URMemObj->Mem).getPtr(Device);
      OL_RETURN_ON_ERR(olMemcpy(nullptr, Ptr, Device->OffloadDevice, HostPtr,
                                Adapter->HostDevice, size, nullptr));
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
    for (auto *Ptr : BufferImpl.Ptrs) {
      if (Ptr) {
        OL_RETURN_ON_ERR(olMemFree(Ptr));
      }
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
