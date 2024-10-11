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
    // TODO: use UMF
    ZeStruct<ze_host_mem_alloc_desc_t> hostDesc;
    ZE2UR_CALL_THROWS(zeMemAllocHost, (hContext->getZeHandle(), &hostDesc, size,
                                       0, &this->ptr));

    if (hostPtr) {
      std::memcpy(this->ptr, hostPtr, size);
    }
  }
}

ur_host_mem_handle_t::~ur_host_mem_handle_t() {
  // TODO: use UMF API here
  if (ptr) {
    ZE_CALL_NOCHECK(zeMemFree, (hContext->getZeHandle(), ptr));
  }
}

void *ur_host_mem_handle_t::getPtr(ur_device_handle_t hDevice) {
  std::ignore = hDevice;
  return ptr;
}

ur_device_mem_handle_t::ur_device_mem_handle_t(ur_context_handle_t hContext,
                                               void *hostPtr, size_t size)
    : ur_mem_handle_t_(hContext, size),
      deviceAllocations(hContext->getPlatform()->getNumDevices()) {
  // Legacy adapter allocated the memory directly on a device (first on the
  // contxt) and if the buffer is used on another device, memory is migrated
  // (depending on an env var setting).
  //
  // TODO: port this behavior or figure out if it makes sense to keep the memory
  // in a host buffer (e.g. for smaller sizes).
  if (hostPtr) {
    buffer.assign(reinterpret_cast<char *>(hostPtr),
                  reinterpret_cast<char *>(hostPtr) + size);
  }
}

ur_device_mem_handle_t::~ur_device_mem_handle_t() {
  // TODO: use UMF API here
  for (auto &ptr : deviceAllocations) {
    if (ptr) {
      ZE_CALL_NOCHECK(zeMemFree, (hContext->getZeHandle(), ptr));
    }
  }
}

void *ur_device_mem_handle_t::getPtr(ur_device_handle_t hDevice) {
  std::lock_guard lock(this->Mutex);

  auto &ptr = deviceAllocations[hDevice->Id.value()];
  if (!ptr) {
    ZeStruct<ze_device_mem_alloc_desc_t> deviceDesc;
    ZE2UR_CALL_THROWS(zeMemAllocDevice, (hContext->getZeHandle(), &deviceDesc,
                                         size, 0, hDevice->ZeDevice, &ptr));

    if (!buffer.empty()) {
      auto commandList = hContext->commandListCache.getImmediateCommandList(
          hDevice->ZeDevice, true,
          hDevice
              ->QueueGroup
                  [ur_device_handle_t_::queue_group_info_t::type::Compute]
              .ZeOrdinal,
          ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
          std::nullopt);
      ZE2UR_CALL_THROWS(
          zeCommandListAppendMemoryCopy,
          (commandList.get(), ptr, buffer.data(), size, nullptr, 0, nullptr));
    }
  }
  return ptr;
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
