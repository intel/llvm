//===--------- memory.cpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ur_interface_loader.hpp"
#include "context.hpp"
#include "memory.hpp"

#include "../helpers/memory_helpers.hpp"
#include "../image_common.hpp"

static bool isAccessCompatible(ur_mem_buffer_t::device_access_mode_t requested,
                               ur_mem_buffer_t::device_access_mode_t actual) {
  return requested == actual ||
         actual == ur_mem_buffer_t::device_access_mode_t::read_write;
}

ur_mem_buffer_t::ur_mem_buffer_t(ur_context_handle_t hContext, size_t size,
                                 device_access_mode_t accessMode)
    : hContext(hContext), size(size), accessMode(accessMode) {}

ur_shared_mutex &ur_mem_buffer_t::getMutex() { return Mutex; }

ur_usm_handle_t::ur_usm_handle_t(ur_context_handle_t hContext, size_t size,
                                 const void *ptr)
    : ur_mem_buffer_t(hContext, size, device_access_mode_t::read_write),
      ptr(const_cast<void *>(ptr)) {}

void *ur_usm_handle_t::getDevicePtr(ur_device_handle_t /*hDevice*/,
                                    device_access_mode_t /*access*/,
                                    size_t offset, size_t /*size*/,
                                    ze_command_list_handle_t /*cmdList*/,
                                    wait_list_view & /*waitListView*/) {
  return ur_cast<char *>(ptr) + offset;
}

void *ur_usm_handle_t::mapHostPtr(ur_map_flags_t /*flags*/, size_t offset,
                                  size_t /*size*/,
                                  ze_command_list_handle_t /*cmdList*/,
                                  wait_list_view & /*waitListView*/) {
  return ur_cast<char *>(ptr) + offset;
}

void ur_usm_handle_t::unmapHostPtr(void * /*pMappedPtr*/,
                                   ze_command_list_handle_t /*cmdList*/,
                                   wait_list_view & /*waitListView*/) {
  /* nop */
}

static v2::raii::command_list_unique_handle
getSyncCommandListForCopy(ur_context_handle_t hContext,
                          ur_device_handle_t hDevice) {
  v2::command_list_desc_t listDesc;
  listDesc.IsInOrder = true;
  listDesc.Ordinal =
      hDevice
          ->QueueGroup[ur_device_handle_t_::queue_group_info_t::type::Compute]
          .ZeOrdinal;
  listDesc.CopyOffloadEnable = true;
  return hContext->getCommandListCache().getImmediateCommandList(
      hDevice->ZeDevice, listDesc, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL, std::nullopt);
}

static ur_result_t synchronousZeCopy(ur_context_handle_t hContext,
                                     ur_device_handle_t hDevice, void *dst,
                                     const void *src, size_t size) try {
  auto commandList = getSyncCommandListForCopy(hContext, hDevice);

  ZE2UR_CALL(zeCommandListAppendMemoryCopy,
             (commandList.get(), dst, src, size, nullptr, 0, nullptr));

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_integrated_buffer_handle_t::ur_integrated_buffer_handle_t(
    ur_context_handle_t hContext, void *hostPtr, size_t size,
    device_access_mode_t accessMode)
    : ur_mem_buffer_t(hContext, size, accessMode) {
  bool hostPtrImported =
      maybeImportUSM(hContext->getPlatform()->ZeDriverHandleExpTranslated,
                     hContext->getZeHandle(), hostPtr, size);

  if (hostPtrImported) {
    this->ptr = usm_unique_ptr_t(hostPtr, [hContext](void *ptr) {
      ZeUSMImport.doZeUSMRelease(
          hContext->getPlatform()->ZeDriverHandleExpTranslated, ptr);
    });
  } else {
    void *rawPtr;
    // Use HOST memory for integrated GPUs to enable zero-copy device access
    UR_CALL_THROWS(hContext->getDefaultUSMPool()->allocate(
        hContext, nullptr, nullptr, UR_USM_TYPE_HOST, size, &rawPtr));

    this->ptr = usm_unique_ptr_t(rawPtr, [hContext](void *ptr) {
      auto ret = hContext->getDefaultUSMPool()->free(ptr);
      if (ret != UR_RESULT_SUCCESS) {
        UR_LOG(ERR, "Failed to free host memory: {}", ret);
      }
    });

    if (hostPtr) {
      // Initial copy using Level Zero for USM HOST memory
      auto hDevice = hContext->getDevices()[0];
      UR_CALL_THROWS(
          synchronousZeCopy(hContext, hDevice, this->ptr.get(), hostPtr, size));
      // Set writeBackPtr to enable map/unmap copy-back (but NOT destructor
      // copy-back)
      writeBackPtr = hostPtr;
    }
  }
}

ur_integrated_buffer_handle_t::ur_integrated_buffer_handle_t(
    ur_context_handle_t hContext, void *hostPtr, size_t size,
    device_access_mode_t accessMode, bool ownHostPtr)
    : ur_mem_buffer_t(hContext, size, accessMode) {
  this->ptr = usm_unique_ptr_t(hostPtr, [hContext, ownHostPtr](void *ptr) {
    if (!ownHostPtr || !checkL0LoaderTeardown()) {
      return;
    }
    ZE_CALL_NOCHECK(zeMemFree, (hContext->getZeHandle(), ptr));
  });
}

void *ur_integrated_buffer_handle_t::getDevicePtr(
    ur_device_handle_t /*hDevice*/, device_access_mode_t /*access*/,
    size_t offset, size_t /*size*/, ze_command_list_handle_t /*cmdList*/,
    wait_list_view & /*waitListView*/) {
  return ur_cast<char *>(ptr.get()) + offset;
}

void *ur_integrated_buffer_handle_t::mapHostPtr(
    ur_map_flags_t flags, size_t offset, size_t mapSize,
    ze_command_list_handle_t /*cmdList*/, wait_list_view & /*waitListView*/) {
  if (writeBackPtr) {
    // Copy-back path: user gets back their original pointer
    void *mappedPtr = ur_cast<char *>(writeBackPtr) + offset;

    if (flags & UR_MAP_FLAG_READ) {
      // Use Level Zero copy for USM HOST memory to ensure GPU visibility
      auto hDevice = hContext->getDevices()[0];
      UR_CALL_THROWS(synchronousZeCopy(hContext, hDevice, mappedPtr,
                                       ur_cast<char *>(ptr.get()) + offset,
                                       mapSize));
    }

    // Track this mapping for unmap
    mappedRegions.emplace_back(usm_unique_ptr_t(mappedPtr, [](void *) {}),
                               mapSize, offset, flags);

    return mappedPtr;
  }

  // Zero-copy path: for successfully imported or USM pointers
  return ur_cast<char *>(ptr.get()) + offset;
}

void ur_integrated_buffer_handle_t::unmapHostPtr(
    void *pMappedPtr, ze_command_list_handle_t /*cmdList*/,
    wait_list_view & /*waitListView*/) {
  if (writeBackPtr) {
    // Copy-back path: find the mapped region and copy data back if needed
    auto mappedRegion =
        std::find_if(mappedRegions.begin(), mappedRegions.end(),
                     [pMappedPtr](const host_allocation_desc_t &desc) {
                       return desc.ptr.get() == pMappedPtr;
                     });

    if (mappedRegion == mappedRegions.end()) {
      UR_DFAILURE("could not find pMappedPtr:" << pMappedPtr);
      throw UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (mappedRegion->flags &
        (UR_MAP_FLAG_WRITE | UR_MAP_FLAG_WRITE_INVALIDATE_REGION)) {
      // Use Level Zero copy for USM HOST memory to ensure GPU visibility
      auto hDevice = hContext->getDevices()[0];
      UR_CALL_THROWS(synchronousZeCopy(
          hContext, hDevice, ur_cast<char *>(ptr.get()) + mappedRegion->offset,
          mappedRegion->ptr.get(), mappedRegion->size));
    }

    mappedRegions.erase(mappedRegion);
    return;
  }
  // No op for zero-copy path, memory is synced
}

ur_integrated_buffer_handle_t::~ur_integrated_buffer_handle_t() {
  // Do NOT do automatic copy-back in destructor - it causes heap corruption
  // because writeBackPtr may be freed by SYCL runtime before buffer destructor
  // runs. Copy-back happens via explicit map/unmap operations (see
  // mapHostPtr/unmapHostPtr).
}

void *ur_discrete_buffer_handle_t::allocateOnDevice(ur_device_handle_t hDevice,
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
          UR_LOG(ERR, "Failed to free device memory: {}", ret);
        }
      });

  activeAllocationDevice = hDevice;

  return ptr;
}

ur_result_t
ur_discrete_buffer_handle_t::migrateBufferTo(ur_device_handle_t hDevice,
                                             void *src, size_t size) {
  TRACK_SCOPE_LATENCY("ur_discrete_buffer_handle_t::migrateBufferTo");

  auto Id = hDevice->Id.value();
  void *dst = deviceAllocations[Id].get() ? deviceAllocations[Id].get()
                                          : allocateOnDevice(hDevice, size);

  UR_CALL(synchronousZeCopy(hContext, hDevice, dst, src, size));

  return UR_RESULT_SUCCESS;
}

ur_discrete_buffer_handle_t::ur_discrete_buffer_handle_t(
    ur_context_handle_t hContext, void *hostPtr, size_t size,
    device_access_mode_t accessMode)
    : ur_mem_buffer_t(hContext, size, accessMode),
      deviceAllocations(hContext->getPlatform()->getNumDevices()),
      activeAllocationDevice(nullptr), mapToPtr(nullptr, nullptr),
      hostAllocations() {
  if (hostPtr) {
    // Try importing the pointer to speed up memory copies for map/unmap
    bool hostPtrImported =
        maybeImportUSM(hContext->getPlatform()->ZeDriverHandleExpTranslated,
                       hContext->getZeHandle(), hostPtr, size);

    if (hostPtrImported) {
      mapToPtr = usm_unique_ptr_t(hostPtr, [hContext](void *ptr) {
        ZeUSMImport.doZeUSMRelease(
            hContext->getPlatform()->ZeDriverHandleExpTranslated, ptr);
      });
    } else {
      mapToPtr = usm_unique_ptr_t(hostPtr, [](void *) {});
    }

    auto initialDevice = hContext->getDevices()[0];
    UR_CALL_THROWS(migrateBufferTo(initialDevice, hostPtr, size));
  }
}

ur_discrete_buffer_handle_t::ur_discrete_buffer_handle_t(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, void *devicePtr,
    size_t size, device_access_mode_t accessMode, void *hostPtr, bool ownZePtr)
    : ur_mem_buffer_t(hContext, size, accessMode),
      deviceAllocations(hContext->getPlatform()->getNumDevices()),
      activeAllocationDevice(hDevice), writeBackPtr(hostPtr),
      hostAllocations() {

  if (!devicePtr) {
    hDevice = hDevice ? hDevice : hContext->getDevices()[0];
    devicePtr = allocateOnDevice(hDevice, size);

    if (hostPtr) {
      UR_CALL_THROWS(migrateBufferTo(hDevice, hostPtr, size));
    }
  } else {
    assert(hDevice);
    deviceAllocations[hDevice->Id.value()] = usm_unique_ptr_t(
        devicePtr, [hContext = this->hContext, ownZePtr](void *ptr) {
          if (!ownZePtr || !checkL0LoaderTeardown()) {
            return;
          }
          ZE_CALL_NOCHECK(zeMemFree, (hContext->getZeHandle(), ptr));
        });
  }
}

ur_discrete_buffer_handle_t::~ur_discrete_buffer_handle_t() {
  if (!activeAllocationDevice || !writeBackPtr)
    return;

  auto srcPtr = getActiveDeviceAlloc();
  synchronousZeCopy(hContext, activeAllocationDevice, writeBackPtr, srcPtr,
                    getSize());
}

void *ur_discrete_buffer_handle_t::getActiveDeviceAlloc(size_t offset) {
  assert(activeAllocationDevice);
  return ur_cast<char *>(
             deviceAllocations[activeAllocationDevice->Id.value()].get()) +
         offset;
}

void *ur_discrete_buffer_handle_t::getDevicePtr(
    ur_device_handle_t hDevice, device_access_mode_t /*access*/, size_t offset,
    size_t /*size*/, ze_command_list_handle_t /*cmdList*/,
    wait_list_view & /*waitListView*/) {
  TRACK_SCOPE_LATENCY("ur_discrete_buffer_handle_t::getDevicePtr");

  if (!activeAllocationDevice) {
    if (!hDevice) {
      hDevice = hContext->getDevices()[0];
    }

    allocateOnDevice(hDevice, getSize());
  }

  if (!hDevice) {
    hDevice = activeAllocationDevice;
  }

  if (activeAllocationDevice == hDevice) {
    return getActiveDeviceAlloc(offset);
  }

  auto &p2pDevices = hContext->getP2PDevices(hDevice);
  auto p2pAccessible = std::find(p2pDevices.begin(), p2pDevices.end(),
                                 activeAllocationDevice) != p2pDevices.end();

  if (!p2pAccessible) {
    // TODO: migrate buffer through the host
    throw UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // TODO: see if it's better to migrate the memory to the specified device
  return getActiveDeviceAlloc(offset);
}

static void migrateMemory(ze_command_list_handle_t cmdList, void *src,
                          void *dst, size_t size,
                          wait_list_view &waitListView) {
  if (!cmdList) {
    UR_DFAILURE("invalid handle in migrateMemory");
    throw UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  }
  ZE2UR_CALL_THROWS(zeCommandListAppendMemoryCopy,
                    (cmdList, dst, src, size, nullptr, waitListView.num,
                     waitListView.handles));
  waitListView.clear();
}

void *ur_discrete_buffer_handle_t::mapHostPtr(ur_map_flags_t flags,
                                              size_t offset, size_t size,
                                              ze_command_list_handle_t cmdList,
                                              wait_list_view &waitListView) {
  TRACK_SCOPE_LATENCY("ur_discrete_buffer_handle_t::mapHostPtr");
  // TODO: use async alloc?

  void *ptr = mapToPtr.get();
  if (!ptr) {
    UR_CALL_THROWS(hContext->getDefaultUSMPool()->allocate(
        hContext, nullptr, nullptr, UR_USM_TYPE_HOST, size, &ptr));
  }

  usm_unique_ptr_t mappedPtr =
      usm_unique_ptr_t(ptr, [ownsAlloc = !bool(mapToPtr), this](void *p) {
        if (ownsAlloc) {
          auto ret = hContext->getDefaultUSMPool()->free(p);
          if (ret != UR_RESULT_SUCCESS) {
            UR_LOG(ERR, "Failed to free mapped memory: {}", ret);
          }
        }
      });

  hostAllocations.emplace_back(std::move(mappedPtr), size, offset, flags);

  if (activeAllocationDevice && (flags & UR_MAP_FLAG_READ)) {
    auto srcPtr = getActiveDeviceAlloc(offset);
    migrateMemory(cmdList, srcPtr, hostAllocations.back().ptr.get(), size,
                  waitListView);
  }

  return hostAllocations.back().ptr.get();
}

void ur_discrete_buffer_handle_t::unmapHostPtr(void *pMappedPtr,
                                               ze_command_list_handle_t cmdList,
                                               wait_list_view &waitListView) {
  TRACK_SCOPE_LATENCY("ur_discrete_buffer_handle_t::unmapHostPtr");

  auto hostAlloc =
      std::find_if(hostAllocations.begin(), hostAllocations.end(),
                   [pMappedPtr](const host_allocation_desc_t &desc) {
                     return desc.ptr.get() == pMappedPtr;
                   });

  if (hostAlloc == hostAllocations.end()) {
    UR_DFAILURE("could not find pMappedPtr:" << pMappedPtr);
    throw UR_RESULT_ERROR_INVALID_ARGUMENT;
  }

  bool shouldMigrateToDevice =
      !(hostAlloc->flags & UR_MAP_FLAG_WRITE_INVALIDATE_REGION);

  if (!activeAllocationDevice && shouldMigrateToDevice) {
    allocateOnDevice(hContext->getDevices()[0], getSize());
  }

  // TODO: tests require that memory is migrated even for
  // UR_MAP_FLAG_WRITE_INVALIDATE_REGION when there is an active device
  // allocation. is this correct?
  if (activeAllocationDevice) {
    migrateMemory(cmdList, hostAlloc->ptr.get(),
                  getActiveDeviceAlloc(hostAlloc->offset), hostAlloc->size,
                  waitListView);
  }

  hostAllocations.erase(hostAlloc);
}

ur_shared_buffer_handle_t::ur_shared_buffer_handle_t(
    ur_context_handle_t hContext, void *sharedPtr, size_t size,
    device_access_mode_t accesMode, bool ownDevicePtr)
    : ur_mem_buffer_t(hContext, size, accesMode),
      ptr(sharedPtr, [hContext, ownDevicePtr](void *ptr) {
        if (!ownDevicePtr || !checkL0LoaderTeardown()) {
          return;
        }
        ZE_CALL_NOCHECK(zeMemFree, (hContext->getZeHandle(), ptr));
      }) {}

void *ur_shared_buffer_handle_t::getDevicePtr(
    ur_device_handle_t, device_access_mode_t, size_t offset, size_t,
    ze_command_list_handle_t /*cmdList*/, wait_list_view & /*waitListView*/) {
  return reinterpret_cast<char *>(ptr.get()) + offset;
}

void *
ur_shared_buffer_handle_t::mapHostPtr(ur_map_flags_t, size_t offset, size_t,
                                      ze_command_list_handle_t /*cmdList*/,
                                      wait_list_view & /*waitListView*/) {
  return reinterpret_cast<char *>(ptr.get()) + offset;
}

void ur_shared_buffer_handle_t::unmapHostPtr(
    void *, ze_command_list_handle_t /*cmdList*/,
    wait_list_view & /*waitListView*/) {
  // nop
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

ur_mem_sub_buffer_t::ur_mem_sub_buffer_t(ur_mem_handle_t hParent, size_t offset,
                                         size_t size,
                                         device_access_mode_t accessMode)
    : ur_mem_buffer_t(hParent->getBuffer()->getContext(), size, accessMode),
      hParent(hParent), offset(offset) {
  ur::level_zero::urMemRetain(hParent);
}

ur_mem_sub_buffer_t::~ur_mem_sub_buffer_t() {
  ur::level_zero::urMemRelease(hParent);
}

void *ur_mem_sub_buffer_t::getDevicePtr(ur_device_handle_t hDevice,
                                        device_access_mode_t access,
                                        size_t offset, size_t size,
                                        ze_command_list_handle_t cmdList,
                                        wait_list_view &waitListView) {
  return hParent->getBuffer()->getDevicePtr(
      hDevice, access, offset + this->offset, size, cmdList, waitListView);
}

void *ur_mem_sub_buffer_t::mapHostPtr(ur_map_flags_t flags, size_t offset,
                                      size_t size,
                                      ze_command_list_handle_t cmdList,
                                      wait_list_view &waitListView) {
  return hParent->getBuffer()->mapHostPtr(flags, offset + this->offset, size,
                                          cmdList, waitListView);
}

void ur_mem_sub_buffer_t::unmapHostPtr(void *pMappedPtr,
                                       ze_command_list_handle_t cmdList,
                                       wait_list_view &waitListView) {
  return hParent->getBuffer()->unmapHostPtr(pMappedPtr, cmdList, waitListView);
}

ur_shared_mutex &ur_mem_sub_buffer_t::getMutex() {
  return hParent->getBuffer()->getMutex();
}

ur_mem_image_t::ur_mem_image_t(ur_context_handle_t hContext,
                               ur_mem_flags_t flags,
                               const ur_image_format_t *pImageFormat,
                               const ur_image_desc_t *pImageDesc, void *pHost)
    : hContext(hContext) {
  UR_CALL_THROWS(ur2zeImageDesc(pImageFormat, pImageDesc, zeImageDesc));

  // Currently we have the "0" device in context with mutliple root devices to
  // own the image.
  // TODO: Implement explicit copying for acessing the image from other devices
  // in the context.
  ur_device_handle_t hDevice = hContext->getDevices()[0];
  ZE2UR_CALL_THROWS(zeImageCreate, (hContext->getZeHandle(), hDevice->ZeDevice,
                                    &zeImageDesc, zeImage.ptr()));

  if ((flags & UR_MEM_FLAG_USE_HOST_POINTER) != 0 ||
      (flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) != 0) {
    // Initialize image synchronously with immediate offload.

    auto commandList = getSyncCommandListForCopy(hContext, hDevice);
    ZE2UR_CALL_THROWS(zeCommandListAppendImageCopyFromMemory,
                      (commandList.get(), zeImage.get(), pHost, nullptr,
                       nullptr, 0, nullptr));
  }
}

ur_mem_image_t::ur_mem_image_t(ur_context_handle_t hContext,
                               const ur_image_format_t *pImageFormat,
                               const ur_image_desc_t *pImageDesc,
                               ze_image_handle_t zeImage, bool ownZeImage)
    : hContext(hContext), zeImage(zeImage, ownZeImage) {
  UR_CALL_THROWS(ur2zeImageDesc(pImageFormat, pImageDesc, zeImageDesc));
}

static void verifyImageRegion([[maybe_unused]] ze_image_desc_t &zeImageDesc,
                              ze_image_region_t &zeRegion, size_t rowPitch,
                              size_t slicePitch) {
#ifndef NDEBUG
  if (!(rowPitch == 0 ||
        // special case RGBA image pitch equal to region's width
        (zeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
         rowPitch == 4 * 4 * zeRegion.width) ||
        (zeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
         rowPitch == 4 * 2 * zeRegion.width) ||
        (zeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
         rowPitch == 4 * zeRegion.width))) {
    UR_DFAILURE("image size is invalid");
    throw UR_RESULT_ERROR_INVALID_IMAGE_SIZE;
  }
#endif
  if (!(slicePitch == 0 || slicePitch == rowPitch * zeRegion.height)) {
    UR_DFAILURE("image size is invalid");
    throw UR_RESULT_ERROR_INVALID_IMAGE_SIZE;
  }
}

std::pair<ze_image_handle_t, ze_image_region_t>
ur_mem_image_t::getRWRegion(ur_rect_offset_t &origin, ur_rect_region_t &region,
                            size_t rowPitch, size_t slicePitch) {
  ze_image_region_t zeRegion;
  UR_CALL_THROWS(getImageRegionHelper(zeImageDesc, &origin, &region, zeRegion));

  verifyImageRegion(zeImageDesc, zeRegion, rowPitch, slicePitch);

  return {zeImage.get(), zeRegion};
}

ur_mem_image_t::copy_desc_t ur_mem_image_t::getCopyRegions(
    ur_mem_image_t &src, ur_mem_image_t &dst, ur_rect_offset_t &srcOrigin,
    ur_rect_offset_t &dstOrigin, ur_rect_region_t &region) {
  ze_image_region_t zeSrcRegion;
  UR_CALL_THROWS(
      getImageRegionHelper(src.zeImageDesc, &srcOrigin, &region, zeSrcRegion));

  ze_image_region_t zeDstRegion;
  UR_CALL_THROWS(
      getImageRegionHelper(dst.zeImageDesc, &dstOrigin, &region, zeDstRegion));

  return {{src.zeImage.get(), zeSrcRegion}, {src.zeImage.get(), zeDstRegion}};
}

namespace ur::level_zero {
ur_result_t urMemBufferCreate(ur_context_handle_t hContext,
                              ur_mem_flags_t flags, size_t size,
                              const ur_buffer_properties_t *pProperties,
                              ur_mem_handle_t *phBuffer) try {
  if (flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    // TODO:
    // Having PI_MEM_FLAGS_HOST_PTR_ALLOC for buffer requires allocation of
    // pinned host memory, see:
    // sycl/doc/extensions/supported/sycl_ext_oneapi_use_pinned_host_memory_property.asciidoc
    // We are however missing such functionality in Level Zero, so we just
    // ignore the flag for now.
  }

  if (flags & UR_MEM_FLAG_USE_HOST_POINTER) {
    // To speed up copies, we always import the host ptr to USM memory
  }

  void *hostPtr = pProperties ? pProperties->pHost : nullptr;
  auto accessMode = ur_mem_buffer_t::getDeviceAccessMode(flags);

  // For integrated devices, use zero-copy host buffers. The integrated buffer
  // constructor will handle all cases:
  // 1. No host pointer - allocate USM host memory
  // 2. Host pointer is already USM - use directly
  // 3. Host pointer can be imported - import it
  // 4. Otherwise - allocate USM and copy-back on destruction
  if (useHostBuffer(hContext)) {
    *phBuffer = ur_mem_handle_t_::create<ur_integrated_buffer_handle_t>(
        hContext, hostPtr, size, accessMode);
  } else {
    *phBuffer = ur_mem_handle_t_::create<ur_discrete_buffer_handle_t>(
        hContext, hostPtr, size, accessMode);
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemBufferPartition(ur_mem_handle_t hMem, ur_mem_flags_t flags,
                                 ur_buffer_create_type_t bufferCreateType,
                                 const ur_buffer_region_t *pRegion,
                                 ur_mem_handle_t *phMem) try {
  auto hBuffer = hMem->getBuffer();

  UR_ASSERT(bufferCreateType == UR_BUFFER_CREATE_TYPE_REGION,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  UR_ASSERT((pRegion->origin < hBuffer->getSize() &&
             pRegion->size <= hBuffer->getSize()),
            UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  auto accessMode = ur_mem_buffer_t::getDeviceAccessMode(flags);

  UR_ASSERT(isAccessCompatible(accessMode, hBuffer->getDeviceAccessMode()),
            UR_RESULT_ERROR_INVALID_VALUE);

  *phMem = ur_mem_handle_t_::create<ur_mem_sub_buffer_t>(
      hMem, pRegion->origin, pRegion->size, accessMode);

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemBufferCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) try {
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

  // assume read-write
  auto accessMode = ur_mem_buffer_t::device_access_mode_t::read_write;

  if (useHostBuffer(hContext) && memoryAttrs.type == ZE_MEMORY_TYPE_HOST) {
    *phMem = ur_mem_handle_t_::create<ur_integrated_buffer_handle_t>(
        hContext, ptr, size, accessMode, ownNativeHandle);
    // if useHostBuffer(hContext) is true but the allocation is on device, we'll
    // treat it as discrete memory
  } else if (memoryAttrs.type == ZE_MEMORY_TYPE_SHARED) {
    // For shared allocation, we can use it directly
    *phMem = ur_mem_handle_t_::create<ur_shared_buffer_handle_t>(
        hContext, ptr, size, accessMode, ownNativeHandle);
  } else {
    if (memoryAttrs.type == ZE_MEMORY_TYPE_HOST) {
      // For host allocation, we need to copy the data to a device buffer
      // and then copy it back on release
      *phMem = ur_mem_handle_t_::create<ur_discrete_buffer_handle_t>(
          hContext, hDevice, nullptr, size, accessMode, ptr, ownNativeHandle);
    } else {
      // For device allocation, we can use it directly
      assert(hDevice);
      *phMem = ur_mem_handle_t_::create<ur_discrete_buffer_handle_t>(
          hContext, hDevice, ptr, size, accessMode, nullptr, ownNativeHandle);
    }
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemGetInfo(ur_mem_handle_t hMem, ur_mem_info_t propName,
                         size_t propSize, void *pPropValue,
                         size_t *pPropSizeRet) try {
  // No locking needed here, we only read const members

  UrReturnHelper returnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_MEM_INFO_CONTEXT: {
    if (hMem->isImage()) {
      return returnValue(hMem->getImage()->getContext());
    } else {
      return returnValue(hMem->getBuffer()->getContext());
    }
  }
  case UR_MEM_INFO_SIZE: {
    if (hMem->isImage()) {
      // TODO: implement size calculation
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    // Get size of the allocation
    return returnValue(size_t{hMem->getBuffer()->getSize()});
  }
  case UR_MEM_INFO_REFERENCE_COUNT: {
    return returnValue(hMem->RefCount.getCount());
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemRetain(ur_mem_handle_t hMem) try {
  hMem->RefCount.retain();
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemRelease(ur_mem_handle_t hMem) try {
  if (!hMem->RefCount.release())
    return UR_RESULT_SUCCESS;

  delete hMem;
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemGetNativeHandle(ur_mem_handle_t hMem,
                                 ur_device_handle_t hDevice,
                                 ur_native_handle_t *phNativeMem) try {
  if (hMem->isImage()) {
    auto hImage = hMem->getImage();
    *phNativeMem = reinterpret_cast<ur_native_handle_t>(hImage->getZeImage());
    return UR_RESULT_SUCCESS;
  }

  auto hBuffer = hMem->getBuffer();

  std::scoped_lock<ur_shared_mutex> lock(hBuffer->getMutex());

  wait_list_view emptyWaitListView(nullptr, 0);
  auto ptr = hBuffer->getDevicePtr(
      hDevice, ur_mem_buffer_t::device_access_mode_t::read_write, 0,
      hBuffer->getSize(), nullptr, emptyWaitListView);
  *phNativeMem = reinterpret_cast<ur_native_handle_t>(ptr);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemImageCreate(ur_context_handle_t hContext, ur_mem_flags_t flags,
                             const ur_image_format_t *pImageFormat,
                             const ur_image_desc_t *pImageDesc, void *pHost,
                             ur_mem_handle_t *phMem) try {
  // TODO: implement read-only, write-only
  if ((flags & UR_MEM_FLAG_READ_WRITE) == 0) {
    die("urMemImageCreate: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  *phMem = ur_mem_handle_t_::create<ur_mem_image_t>(
      hContext, flags, pImageFormat, pImageDesc, pHost);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemImageCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) try {
  auto zeImage = reinterpret_cast<ze_image_handle_t>(hNativeMem);
  bool ownNativeHandle = pProperties ? pProperties->isNativeHandleOwned : false;

  *phMem = ur_mem_handle_t_::create<ur_mem_image_t>(
      hContext, pImageFormat, pImageDesc, zeImage, ownNativeHandle);
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urMemImageGetInfo(ur_mem_handle_t /*hMemory*/,
                              ur_image_info_t /*propName*/, size_t /*propSize*/,
                              void * /*pPropValue*/,
                              size_t * /*pPropSizeRet*/) {
  UR_LOG(ERR, "{} function not implemented!", __FUNCTION__);

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCGetMemHandleExp(ur_context_handle_t, void *pMem,
                                 void **ppIPCMemHandleData,
                                 size_t *pIPCMemHandleDataSizeRet) {
  umf_memory_pool_handle_t umfPool;
  auto urRet = umf::umf2urResult(umfPoolByPtr(pMem, &umfPool));
  if (urRet)
    return urRet;

  // Fast path for returning the size of the handle only.
  if (!ppIPCMemHandleData)
    return umf::umf2urResult(
        umfPoolGetIPCHandleSize(umfPool, pIPCMemHandleDataSizeRet));

  size_t fallbackUMFHandleSize = 0;
  size_t *umfHandleSize = pIPCMemHandleDataSizeRet != nullptr
                              ? pIPCMemHandleDataSizeRet
                              : &fallbackUMFHandleSize;
  return umf::umf2urResult(umfGetIPCHandle(
      pMem, reinterpret_cast<umf_ipc_handle_t *>(ppIPCMemHandleData),
      umfHandleSize));
}

ur_result_t urIPCPutMemHandleExp(ur_context_handle_t, void *pIPCMemHandleData) {
  return umf::umf2urResult(
      umfPutIPCHandle(reinterpret_cast<umf_ipc_handle_t>(pIPCMemHandleData)));
}

ur_result_t urIPCOpenMemHandleExp(ur_context_handle_t hContext,
                                  ur_device_handle_t hDevice,
                                  void *pIPCMemHandleData,
                                  size_t ipcMemHandleDataSize, void **ppMem) {
  auto *pool = hContext->getDefaultUSMPool()->getPool(
      usm::pool_descriptor{hContext->getDefaultUSMPool(), hContext, hDevice,
                           UR_USM_TYPE_DEVICE, false});
  if (!pool)
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  umf_memory_pool_handle_t umfPool = pool->umfPool.get();

  size_t umfHandleSize = 0;
  auto urRet =
      umf::umf2urResult(umfPoolGetIPCHandleSize(umfPool, &umfHandleSize));
  if (urRet)
    return urRet;

  if (umfHandleSize != ipcMemHandleDataSize)
    return UR_RESULT_ERROR_INVALID_VALUE;

  umf_ipc_handler_handle_t umfIPCHandler;
  urRet = umf::umf2urResult(umfPoolGetIPCHandler(umfPool, &umfIPCHandler));
  if (urRet)
    return urRet;

  return umf::umf2urResult(umfOpenIPCHandle(
      umfIPCHandler, reinterpret_cast<umf_ipc_handle_t>(pIPCMemHandleData),
      ppMem));
}

ur_result_t urIPCCloseMemHandleExp(ur_context_handle_t, void *pMem) {
  return umf::umf2urResult(umfCloseIPCHandle(pMem));
}

} // namespace ur::level_zero
