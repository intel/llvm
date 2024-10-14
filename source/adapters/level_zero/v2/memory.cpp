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

ur_usm_handle_t_::ur_usm_handle_t_(ur_context_handle_t hContext, size_t size,
                                   const void *ptr)
    : ur_mem_handle_t_(hContext, size), ptr(const_cast<void *>(ptr)) {}

ur_usm_handle_t_::~ur_usm_handle_t_() {}

void *ur_usm_handle_t_::getDevicePtr(
    ur_device_handle_t hDevice, access_mode_t access, size_t offset,
    size_t size, std::function<void(void *src, void *dst, size_t)> migrate) {
  std::ignore = hDevice;
  std::ignore = access;
  std::ignore = offset;
  std::ignore = size;
  std::ignore = migrate;
  return ptr;
}

void *ur_usm_handle_t_::mapHostPtr(
    access_mode_t access, size_t offset, size_t size,
    std::function<void(void *src, void *dst, size_t)>) {
  std::ignore = access;
  std::ignore = offset;
  std::ignore = size;
  return ptr;
}

void ur_usm_handle_t_::unmapHostPtr(
    void *pMappedPtr, std::function<void(void *src, void *dst, size_t)>) {
  std::ignore = pMappedPtr;
  /* nop */
}

ur_integrated_mem_handle_t::ur_integrated_mem_handle_t(
    ur_context_handle_t hContext, void *hostPtr, size_t size,
    host_ptr_action_t hostPtrAction)
    : ur_mem_handle_t_(hContext, size) {
  bool hostPtrImported = false;
  if (hostPtrAction == host_ptr_action_t::import) {
    hostPtrImported =
        maybeImportUSM(hContext->getPlatform()->ZeDriverHandleExpTranslated,
                       hContext->getZeHandle(), hostPtr, size);
  }

  if (hostPtrImported) {
    this->ptr = usm_unique_ptr_t(hostPtr, [hContext](void *ptr) {
      ZeUSMImport.doZeUSMRelease(
          hContext->getPlatform()->ZeDriverHandleExpTranslated, ptr);
    });
  } else {
    void *rawPtr;
    UR_CALL_THROWS(hContext->getDefaultUSMPool()->allocate(
        hContext, nullptr, nullptr, UR_USM_TYPE_HOST, size, &rawPtr));

    this->ptr = usm_unique_ptr_t(rawPtr, [hContext](void *ptr) {
      auto ret = hContext->getDefaultUSMPool()->free(ptr);
      if (ret != UR_RESULT_SUCCESS) {
        logger::error("Failed to free host memory: {}", ret);
      }
    });

    if (hostPtr) {
      std::memcpy(this->ptr.get(), hostPtr, size);
    }
  }
}

ur_integrated_mem_handle_t::ur_integrated_mem_handle_t(
    ur_context_handle_t hContext, void *hostPtr, size_t size, bool ownHostPtr)
    : ur_mem_handle_t_(hContext, size) {
  this->ptr = usm_unique_ptr_t(hostPtr, [hContext, ownHostPtr](void *ptr) {
    if (!ownHostPtr) {
      return;
    }
    auto ret = hContext->getDefaultUSMPool()->free(ptr);
    if (ret != UR_RESULT_SUCCESS) {
      logger::error("Failed to free host memory: {}", ret);
    }
  });
}

void *ur_integrated_mem_handle_t::getDevicePtr(
    ur_device_handle_t hDevice, access_mode_t access, size_t offset,
    size_t size, std::function<void(void *src, void *dst, size_t)> migrate) {
  std::ignore = hDevice;
  std::ignore = access;
  std::ignore = offset;
  std::ignore = size;
  std::ignore = migrate;
  return ptr.get();
}

void *ur_integrated_mem_handle_t::mapHostPtr(
    access_mode_t access, size_t offset, size_t size,
    std::function<void(void *src, void *dst, size_t)> migrate) {
  std::ignore = access;
  std::ignore = offset;
  std::ignore = size;
  std::ignore = migrate;
  return ptr.get();
}

void ur_integrated_mem_handle_t::unmapHostPtr(
    void *pMappedPtr, std::function<void(void *src, void *dst, size_t)>) {
  std::ignore = pMappedPtr;
  /* nop */
}

static ur_result_t synchronousZeCopy(ur_context_handle_t hContext,
                                     ur_device_handle_t hDevice, void *dst,
                                     const void *src, size_t size) {
  auto commandList = hContext->commandListCache.getImmediateCommandList(
      hDevice->ZeDevice, true,
      hDevice
          ->QueueGroup[ur_device_handle_t_::queue_group_info_t::type::Compute]
          .ZeOrdinal,
      ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
      std::nullopt);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (commandList.get(), dst, src, size, nullptr, 0, nullptr));

  return UR_RESULT_SUCCESS;
}

void *ur_discrete_mem_handle_t::allocateOnDevice(ur_device_handle_t hDevice,
                                                 size_t size) {
  assert(hDevice);

  auto id = hDevice->Id.value();
  assert(deviceAllocations[id].get() == nullptr);

  void *ptr;
  UR_CALL_THROWS(hContext->getDefaultUSMPool()->allocate(
      hContext, hDevice, nullptr, UR_USM_TYPE_DEVICE, size, &ptr));

  deviceAllocations[id] =
      usm_unique_ptr_t(ptr, [hContext = this->hContext](void *ptr) {
        auto ret = hContext->getDefaultUSMPool()->free(ptr);
        if (ret != UR_RESULT_SUCCESS) {
          logger::error("Failed to free device memory: {}", ret);
        }
      });

  activeAllocationDevice = hDevice;

  return ptr;
}

ur_result_t
ur_discrete_mem_handle_t::migrateBufferTo(ur_device_handle_t hDevice, void *src,
                                          size_t size) {
  TRACK_SCOPE_LATENCY("ur_discrete_mem_handle_t::migrateBufferTo");

  auto Id = hDevice->Id.value();
  void *dst = deviceAllocations[Id].get() ? deviceAllocations[Id].get()
                                          : allocateOnDevice(hDevice, size);

  UR_CALL(synchronousZeCopy(hContext, hDevice, dst, src, size));

  return UR_RESULT_SUCCESS;
}

ur_discrete_mem_handle_t::ur_discrete_mem_handle_t(ur_context_handle_t hContext,
                                                   void *hostPtr, size_t size)
    : ur_mem_handle_t_(hContext, size),
      deviceAllocations(hContext->getPlatform()->getNumDevices()),
      activeAllocationDevice(nullptr), hostAllocations() {
  if (hostPtr) {
    auto initialDevice = hContext->getDevices()[0];
    UR_CALL_THROWS(migrateBufferTo(initialDevice, hostPtr, size));
  }
}

ur_discrete_mem_handle_t::ur_discrete_mem_handle_t(ur_context_handle_t hContext,
                                                   ur_device_handle_t hDevice,
                                                   void *devicePtr, size_t size,
                                                   void *writeBackMemory,
                                                   bool ownZePtr)
    : ur_mem_handle_t_(hContext, size),
      deviceAllocations(hContext->getPlatform()->getNumDevices()),
      activeAllocationDevice(hDevice), writeBackPtr(writeBackMemory),
      hostAllocations() {

  if (!devicePtr) {
    hDevice = hDevice ? hDevice : hContext->getDevices()[0];
    devicePtr = allocateOnDevice(hDevice, size);
  } else {
    deviceAllocations[hDevice->Id.value()] = usm_unique_ptr_t(
        devicePtr, [hContext = this->hContext, ownZePtr](void *ptr) {
          if (!ownZePtr) {
            return;
          }
          auto ret = hContext->getDefaultUSMPool()->free(ptr);
          if (ret != UR_RESULT_SUCCESS) {
            logger::error("Failed to free device memory: {}", ret);
          }
        });
  }
}

ur_discrete_mem_handle_t::~ur_discrete_mem_handle_t() {
  if (!activeAllocationDevice || !writeBackPtr)
    return;

  auto srcPtr = ur_cast<char *>(
      deviceAllocations[activeAllocationDevice->Id.value()].get());
  synchronousZeCopy(hContext, activeAllocationDevice, writeBackPtr, srcPtr,
                    getSize());
}

void *ur_discrete_mem_handle_t::getDevicePtr(
    ur_device_handle_t hDevice, access_mode_t access, size_t offset,
    size_t size, std::function<void(void *src, void *dst, size_t)> migrate) {
  TRACK_SCOPE_LATENCY("ur_discrete_mem_handle_t::getDevicePtr");

  std::ignore = access;
  std::ignore = size;
  std::ignore = migrate;

  if (!activeAllocationDevice) {
    if (!hDevice) {
      hDevice = hContext->getDevices()[0];
    }

    allocateOnDevice(hDevice, getSize());
  }

  char *ptr;
  if (activeAllocationDevice == hDevice) {
    ptr = ur_cast<char *>(deviceAllocations[hDevice->Id.value()].get());
    return ptr + offset;
  }

  auto &p2pDevices = hContext->getP2PDevices(hDevice);
  auto p2pAccessible = std::find(p2pDevices.begin(), p2pDevices.end(),
                                 activeAllocationDevice) != p2pDevices.end();

  if (!p2pAccessible) {
    // TODO: migrate buffer through the host
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // TODO: see if it's better to migrate the memory to the specified device
  return ur_cast<char *>(
             deviceAllocations[activeAllocationDevice->Id.value()].get()) +
         offset;
}

void *ur_discrete_mem_handle_t::mapHostPtr(
    access_mode_t access, size_t offset, size_t size,
    std::function<void(void *src, void *dst, size_t)> migrate) {
  TRACK_SCOPE_LATENCY("ur_discrete_mem_handle_t::mapHostPtr");

  // TODO: use async alloc?

  void *ptr;
  UR_CALL_THROWS(hContext->getDefaultUSMPool()->allocate(
      hContext, nullptr, nullptr, UR_USM_TYPE_HOST, size, &ptr));

  hostAllocations.emplace_back(ptr, size, offset, access);

  if (activeAllocationDevice && access != access_mode_t::write_only) {
    auto srcPtr =
        ur_cast<char *>(
            deviceAllocations[activeAllocationDevice->Id.value()].get()) +
        offset;
    migrate(srcPtr, hostAllocations.back().ptr, size);
  }

  return hostAllocations.back().ptr;
}

void ur_discrete_mem_handle_t::unmapHostPtr(
    void *pMappedPtr,
    std::function<void(void *src, void *dst, size_t)> migrate) {
  TRACK_SCOPE_LATENCY("ur_discrete_mem_handle_t::unmapHostPtr");

  for (auto &hostAllocation : hostAllocations) {
    if (hostAllocation.ptr == pMappedPtr) {
      void *devicePtr = nullptr;
      if (activeAllocationDevice) {
        devicePtr =
            ur_cast<char *>(
                deviceAllocations[activeAllocationDevice->Id.value()].get()) +
            hostAllocation.offset;
      } else if (hostAllocation.access != access_mode_t::write_invalidate) {
        devicePtr = ur_cast<char *>(
            getDevicePtr(hContext->getDevices()[0], access_mode_t::read_only,
                         hostAllocation.offset, hostAllocation.size, migrate));
      }

      if (devicePtr) {
        migrate(hostAllocation.ptr, devicePtr, hostAllocation.size);
      }

      // TODO: use async free here?
      UR_CALL_THROWS(hContext->getDefaultUSMPool()->free(hostAllocation.ptr));
      return;
    }
  }

  // No mapping found
  throw UR_RESULT_ERROR_INVALID_ARGUMENT;
}

static bool useHostBuffer(ur_context_handle_t hContext) {
  // We treat integrated devices (physical memory shared with the CPU)
  // differently from discrete devices (those with distinct memories).
  // For integrated devices, allocating the buffer in the host memory
  // enables automatic access from the device, and makes copying
  // unnecessary in the map/unmap operations. This improves performance.
  return hContext->getDevices().size() == 1 &&
         hContext->getDevices()[0]->ZeDeviceProperties->flags &
             ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;
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

  if (useHostBuffer(hContext)) {
    // TODO: assert that if hostPtr is set, either UR_MEM_FLAG_USE_HOST_POINTER
    // or UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER is set?
    auto hostPtrAction =
        flags & UR_MEM_FLAG_USE_HOST_POINTER
            ? ur_integrated_mem_handle_t::host_ptr_action_t::import
            : ur_integrated_mem_handle_t::host_ptr_action_t::copy;
    *phBuffer =
        new ur_integrated_mem_handle_t(hContext, hostPtr, size, hostPtrAction);
  } else {
    *phBuffer = new ur_discrete_mem_handle_t(hContext, hostPtr, size);
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

  auto ptr = reinterpret_cast<void *>(hNativeMem);
  bool ownNativeHandle = pProperties ? pProperties->isNativeHandleOwned : false;

  // Get base of the allocation
  void *base;
  size_t size;
  ZE2UR_CALL(zeMemGetAddressRange,
             (hContext->getZeHandle(), ptr, &base, &size));
  UR_ASSERT(ptr == base, UR_RESULT_ERROR_INVALID_VALUE);

  ze_device_handle_t zeDevice;
  ZeStruct<ze_memory_allocation_properties_t> memoryAttrs;
  UR_CALL(
      getMemoryAttrs(hContext->getZeHandle(), ptr, &zeDevice, &memoryAttrs));

  if (memoryAttrs.type == ZE_MEMORY_TYPE_UNKNOWN) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  ur_device_handle_t hDevice{};
  if (zeDevice) {
    hDevice = hContext->getPlatform()->getDeviceFromNativeHandle(zeDevice);
    UR_ASSERT(hContext->isValidDevice(hDevice),
              UR_RESULT_ERROR_INVALID_CONTEXT);
  }

  if (useHostBuffer(hContext) && memoryAttrs.type == ZE_MEMORY_TYPE_HOST) {
    *phMem =
        new ur_integrated_mem_handle_t(hContext, ptr, size, ownNativeHandle);
    // if useHostBuffer(hContext) is true but the allocation is on device, we'll
    // treat it as discrete memory
  } else {
    if (memoryAttrs.type == ZE_MEMORY_TYPE_HOST) {
      // For host allocation, we need to copy the data to a device buffer
      // and then copy it back on release
      *phMem = new ur_discrete_mem_handle_t(hContext, hDevice, nullptr, size,
                                            ptr, ownNativeHandle);
    } else {
      // For device/shared allocation, we can use it directly
      *phMem = new ur_discrete_mem_handle_t(hContext, hDevice, ptr, size,
                                            nullptr, ownNativeHandle);
    }
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t urMemGetInfo(ur_mem_handle_t hMemory, ur_mem_info_t propName,
                         size_t propSize, void *pPropValue,
                         size_t *pPropSizeRet) {
  // No locking needed here, we only read const members

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

ur_result_t urMemGetNativeHandle(ur_mem_handle_t hMem,
                                 ur_device_handle_t hDevice,
                                 ur_native_handle_t *phNativeMem) {
  std::ignore = hDevice;

  std::scoped_lock<ur_shared_mutex> lock(hMem->Mutex);

  auto ptr =
      hMem->getDevicePtr(nullptr, ur_mem_handle_t_::access_mode_t::read_write,
                         0, hMem->getSize(), nullptr);
  *phNativeMem = reinterpret_cast<ur_native_handle_t>(ptr);
  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
