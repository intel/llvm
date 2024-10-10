//===--------- memory.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "context.hpp"

#include "../helpers/memory_helpers.hpp"

ur_mem_handle_t_::ur_mem_handle_t_(ur_context_handle_t hContext, size_t size)
    : hContext(hContext), size(size) {}

ur_host_mem_handle_t::ur_host_mem_handle_t(ur_context_handle_t hContext,
                                           void *hostPtr, size_t size,
                                           host_ptr_action_t hostPtrAction)
    : ur_mem_handle_t_(hContext, size) {
  bool hostPtrImported = false;
  if (hostPtrAction == host_ptr_action_t::import) {
    hostPtrImported =
        maybeImportUSM(hContext->getPlatform()->ZeDriverHandleExpTranslated,
                       hContext->getZeHandle(), hostPtr, size);
  }

  if (!hostPtrImported) {
    UR_CALL_THROWS(hContext->getDefaultUSMPool()->allocate(
        hContext, nullptr, nullptr, UR_USM_TYPE_HOST, size, &this->ptr));

    if (hostPtr) {
      std::memcpy(this->ptr, hostPtr, size);
    }
  }
}

ur_host_mem_handle_t::~ur_host_mem_handle_t() {
  if (ptr) {
    auto ret = hContext->getDefaultUSMPool()->free(ptr);
    if (ret != UR_RESULT_SUCCESS) {
      logger::error("Failed to free host memory: {}", ret);
    }
  }
}

void *ur_host_mem_handle_t::getPtr(ur_device_handle_t hDevice) {
  std::ignore = hDevice;
  return ptr;
}

ur_result_t ur_device_mem_handle_t::migrateBufferTo(ur_device_handle_t hDevice,
                                                    void *src, size_t size) {
  auto Id = hDevice->Id.value();

  if (!deviceAllocations[Id]) {
    UR_CALL(hContext->getDefaultUSMPool()->allocate(hContext, hDevice, nullptr,
                                                    UR_USM_TYPE_DEVICE, size,
                                                    &deviceAllocations[Id]));
  }

  auto commandList = hContext->commandListCache.getImmediateCommandList(
      hDevice->ZeDevice, true,
      hDevice
          ->QueueGroup[ur_device_handle_t_::queue_group_info_t::type::Compute]
          .ZeOrdinal,
      ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
      std::nullopt);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (commandList.get(), deviceAllocations[Id], src, size, nullptr, 0,
              nullptr));

  activeAllocationDevice = hDevice;

  return UR_RESULT_SUCCESS;
}

ur_device_mem_handle_t::ur_device_mem_handle_t(ur_context_handle_t hContext,
                                               void *hostPtr, size_t size)
    : ur_mem_handle_t_(hContext, size),
      deviceAllocations(hContext->getPlatform()->getNumDevices()),
      activeAllocationDevice(nullptr) {
  if (hostPtr) {
    auto initialDevice = hContext->getDevices()[0];
    UR_CALL_THROWS(migrateBufferTo(initialDevice, hostPtr, size));
  }
}

ur_device_mem_handle_t::~ur_device_mem_handle_t() {
  for (auto &ptr : deviceAllocations) {
    if (ptr) {
      auto ret = hContext->getDefaultUSMPool()->free(ptr);
      if (ret != UR_RESULT_SUCCESS) {
        logger::error("Failed to free device memory: {}", ret);
      }
    }
  }
}

void *ur_device_mem_handle_t::getPtr(ur_device_handle_t hDevice) {
  std::lock_guard lock(this->Mutex);

  if (!activeAllocationDevice) {
    UR_CALL_THROWS(hContext->getDefaultUSMPool()->allocate(
        hContext, hDevice, nullptr, UR_USM_TYPE_DEVICE, getSize(),
        &deviceAllocations[hDevice->Id.value()]));
    activeAllocationDevice = hDevice;
  }

  if (activeAllocationDevice == hDevice) {
    return deviceAllocations[hDevice->Id.value()];
  }

  auto &p2pDevices = hContext->getP2PDevices(hDevice);
  auto p2pAccessible = std::find(p2pDevices.begin(), p2pDevices.end(),
                                 activeAllocationDevice) != p2pDevices.end();

  if (!p2pAccessible) {
    // TODO: migrate buffer through the host
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // TODO: see if it's better to migrate the memory to the specified device
  return deviceAllocations[activeAllocationDevice->Id.value()];
}

namespace ur::level_zero {
ur_result_t urMemBufferCreate(ur_context_handle_t hContext,
                              ur_mem_flags_t flags, size_t size,
                              const ur_buffer_properties_t *pProperties,
                              ur_mem_handle_t *phBuffer) {
  if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    // TODO:
    // Having PI_MEM_FLAGS_HOST_PTR_ALLOC for buffer requires allocation of
    // pinned host memory, see:
    // sycl/doc/extensions/supported/sycl_ext_oneapi_use_pinned_host_memory_property.asciidoc
    // We are however missing such functionality in Level Zero, so we just
    // ignore the flag for now.
  }

  void *hostPtr = pProperties ? pProperties->pHost : nullptr;

  // We treat integrated devices (physical memory shared with the CPU)
  // differently from discrete devices (those with distinct memories).
  // For integrated devices, allocating the buffer in the host memory
  // enables automatic access from the device, and makes copying
  // unnecessary in the map/unmap operations. This improves performance.
  bool useHostBuffer = hContext->getDevices().size() == 1 &&
                       hContext->getDevices()[0]->ZeDeviceProperties->flags &
                           ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

  if (useHostBuffer) {
    // TODO: assert that if hostPtr is set, either UR_MEM_FLAG_USE_HOST_POINTER
    // or UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER is set?
    auto hostPtrAction = flags & UR_MEM_FLAG_USE_HOST_POINTER
                             ? ur_host_mem_handle_t::host_ptr_action_t::import
                             : ur_host_mem_handle_t::host_ptr_action_t::copy;
    *phBuffer =
        new ur_host_mem_handle_t(hContext, hostPtr, size, hostPtrAction);
  } else {
    *phBuffer = new ur_device_mem_handle_t(hContext, hostPtr, size);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemBufferPartition(ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
                                 ur_buffer_create_type_t bufferCreateType,
                                 const ur_buffer_region_t *pRegion,
                                 ur_mem_handle_t *phMem) {
  std::ignore = hBuffer;
  std::ignore = flags;
  std::ignore = bufferCreateType;
  std::ignore = pRegion;
  std::ignore = phMem;
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urMemBufferCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) {
  std::ignore = hNativeMem;
  std::ignore = hContext;
  std::ignore = pProperties;
  std::ignore = phMem;
  logger::error("{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urMemGetInfo(ur_mem_handle_t hMemory, ur_mem_info_t propName,
                         size_t propSize, void *pPropValue,
                         size_t *pPropSizeRet) {
  std::shared_lock<ur_shared_mutex> Lock(hMemory->Mutex);
  UrReturnHelper returnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_MEM_INFO_CONTEXT: {
    return returnValue(hMemory->getContext());
  }
  case UR_MEM_INFO_SIZE: {
    // Get size of the allocation
    return returnValue(size_t{hMemory->getSize()});
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemRetain(ur_mem_handle_t hMem) {
  hMem->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t urMemRelease(ur_mem_handle_t hMem) {
  if (hMem->RefCount.decrementAndTest()) {
    delete hMem;
  }
  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
