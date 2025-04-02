//===--------- memory.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>

#include <ur_api.h>

#include "../device.hpp"
#include "common.hpp"

using usm_unique_ptr_t = std::unique_ptr<void, std::function<void(void *)>>;

struct ur_mem_buffer_t : _ur_object {
  // Indicates if this object is an interop handle.
  bool IsInteropNativeHandle = false;

  enum class device_access_mode_t { read_write, read_only, write_only };

  ur_mem_buffer_t(ur_context_handle_t hContext, size_t size,
                  device_access_mode_t accesMode);
  virtual ~ur_mem_buffer_t() = default;

  virtual ur_shared_mutex &getMutex();

  // Following functions should always be called under the lock.

  // Returns pointer to the device memory. If device handle is NULL,
  // the buffer is allocated on the first device in the context.
  virtual void *
  getDevicePtr(ur_device_handle_t, device_access_mode_t, size_t offset,
               size_t size,
               std::function<void(void *src, void *dst, size_t)> mecmpy) = 0;
  virtual void *
  mapHostPtr(ur_map_flags_t, size_t offset, size_t size,
             std::function<void(void *src, void *dst, size_t)> memcpy) = 0;
  virtual void
  unmapHostPtr(void *pMappedPtr,
               std::function<void(void *src, void *dst, size_t)> memcpy) = 0;

  device_access_mode_t getDeviceAccessMode() const { return accessMode; }
  ur_context_handle_t getContext() const { return hContext; }
  size_t getSize() const { return size; }

protected:
  const ur_context_handle_t hContext;
  const size_t size;
  const device_access_mode_t accessMode;
};

// non-owning buffer wrapper around USM pointer
struct ur_usm_handle_t : ur_mem_buffer_t {
  ur_usm_handle_t(ur_context_handle_t hContext, size_t size, const void *ptr);

  void *
  getDevicePtr(ur_device_handle_t, device_access_mode_t, size_t offset,
               size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(ur_map_flags_t, size_t offset, size_t size,
                   std::function<void(void *src, void *dst, size_t)>) override;
  void unmapHostPtr(void *pMappedPtr,
                    std::function<void(void *src, void *dst, size_t)>) override;

private:
  void *ptr;
};

// Manages memory buffer for integrated GPU.
// For integrated devices the buffer has been allocated in host memory
// and can be accessed by the device without copying.
struct ur_integrated_buffer_handle_t : ur_mem_buffer_t {
  enum class host_ptr_action_t { import, copy };

  ur_integrated_buffer_handle_t(ur_context_handle_t hContext, void *hostPtr,
                                size_t size, host_ptr_action_t useHostPtr,
                                device_access_mode_t accesMode);

  ur_integrated_buffer_handle_t(ur_context_handle_t hContext, void *hostPtr,
                                size_t size, device_access_mode_t accesMode,
                                bool ownHostPtr, bool interopNativeHandle);

  void *
  getDevicePtr(ur_device_handle_t, device_access_mode_t, size_t offset,
               size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(ur_map_flags_t, size_t offset, size_t size,
                   std::function<void(void *src, void *dst, size_t)>) override;
  void unmapHostPtr(void *pMappedPtr,
                    std::function<void(void *src, void *dst, size_t)>) override;

private:
  usm_unique_ptr_t ptr;
};

struct host_allocation_desc_t {
  host_allocation_desc_t(usm_unique_ptr_t ptr, size_t size, size_t offset,
                         ur_map_flags_t flags)
      : ptr(std::move(ptr)), size(size), offset(offset), flags(flags) {}

  usm_unique_ptr_t ptr;
  size_t size;
  size_t offset;
  ur_map_flags_t flags;
};

// Manages memory buffer for discrete GPU.
// Memory is allocated on the device and migrated/copies if necessary.
struct ur_discrete_buffer_handle_t : ur_mem_buffer_t {
  // If hostPtr is not null, the buffer is allocated immediately on the
  // first device in the context. Otherwise, the buffer is allocated on
  // firt getDevicePtr call.
  ur_discrete_buffer_handle_t(ur_context_handle_t hContext, void *hostPtr,
                              size_t size, device_access_mode_t accesMode);
  ~ur_discrete_buffer_handle_t();

  // Create buffer on top of existing device memory.
  ur_discrete_buffer_handle_t(ur_context_handle_t hContext,
                              ur_device_handle_t hDevice, void *devicePtr,
                              size_t size, device_access_mode_t accesMode,
                              void *writeBackMemory, bool ownDevicePtr,
                              bool interopNativeHandle);

  void *
  getDevicePtr(ur_device_handle_t, device_access_mode_t, size_t offset,
               size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(ur_map_flags_t, size_t offset, size_t size,
                   std::function<void(void *src, void *dst, size_t)>) override;
  void unmapHostPtr(void *pMappedPtr,
                    std::function<void(void *src, void *dst, size_t)>) override;

private:
  void *getCurrentAllocation();

  // Vector of per-device allocations indexed by device->Id
  std::vector<usm_unique_ptr_t> deviceAllocations;

  // Specifies device on which the latest allocation resides.
  // If null, there is no allocation.
  ur_device_handle_t activeAllocationDevice = nullptr;

  // If not null, copy the buffer content back to this memory on release.
  void *writeBackPtr = nullptr;

  // If not null, mapHostPtr should map memory to this ptr
  void *mapToPtr = nullptr;

  std::vector<host_allocation_desc_t> hostAllocations;

  void *getActiveDeviceAlloc(size_t offset = 0);
  void *allocateOnDevice(ur_device_handle_t hDevice, size_t size);
  ur_result_t migrateBufferTo(ur_device_handle_t hDevice, void *src,
                              size_t size);
};

struct ur_shared_buffer_handle_t : ur_mem_buffer_t {
  ur_shared_buffer_handle_t(ur_context_handle_t hContext, void *devicePtr,
                            size_t size, device_access_mode_t accesMode,
                            bool ownDevicePtr);

  void *
  getDevicePtr(ur_device_handle_t, device_access_mode_t, size_t offset,
               size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(ur_map_flags_t, size_t offset, size_t size,
                   std::function<void(void *src, void *dst, size_t)>) override;
  void unmapHostPtr(void *pMappedPtr,
                    std::function<void(void *src, void *dst, size_t)>) override;

private:
  usm_unique_ptr_t ptr;
};

struct ur_mem_sub_buffer_t : ur_mem_buffer_t {
  ur_mem_sub_buffer_t(ur_mem_handle_t hParent, size_t offset, size_t size,
                      device_access_mode_t accesMode);
  ~ur_mem_sub_buffer_t();

  void *
  getDevicePtr(ur_device_handle_t, device_access_mode_t, size_t offset,
               size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(ur_map_flags_t, size_t offset, size_t size,
                   std::function<void(void *src, void *dst, size_t)>) override;
  void unmapHostPtr(void *pMappedPtr,
                    std::function<void(void *src, void *dst, size_t)>) override;

  ur_shared_mutex &getMutex() override;

private:
  ur_mem_handle_t hParent;
  size_t offset;
};

struct ur_mem_image_t : _ur_object {
  ur_mem_image_t(ur_context_handle_t hContext, ur_mem_flags_t flags,
                 const ur_image_format_t *pImageFormat,
                 const ur_image_desc_t *pImageDesc, void *pHost);
  ur_mem_image_t(ur_context_handle_t, const ur_image_format_t *pImageFormat,
                 const ur_image_desc_t *pImageDesc, ze_image_handle_t zeImage,
                 bool ownZeImage, bool interopNativeHandle);

  ze_image_handle_t getZeImage() const { return zeImage.get(); }

  std::pair<ze_image_handle_t, ze_image_region_t>
  getRWRegion(ur_rect_offset_t &origin, ur_rect_region_t &region,
              size_t rowPitch, size_t slicePitch);

  struct copy_desc_t {
    std::pair<ze_image_handle_t, ze_image_region_t> src;
    std::pair<ze_image_handle_t, ze_image_region_t> dst;
  };

  static copy_desc_t getCopyRegions(ur_mem_image_t &src, ur_mem_image_t &dst,
                                    ur_rect_offset_t &srcOrigin,
                                    ur_rect_offset_t &dstOrigin,
                                    ur_rect_region_t &region);

  ur_context_handle_t getContext() const { return hContext; }

private:
  const ur_context_handle_t hContext;
  v2::raii::ze_image_handle_t zeImage;
  ZeStruct<ze_image_desc_t> zeImageDesc;
};

struct ur_mem_handle_t_ {
  template <typename T, typename... Args>
  static ur_mem_handle_t_ *create(Args &&...args) {
    return new ur_mem_handle_t_(std::in_place_type<T>,
                                std::forward<Args>(args)...);
  }

  ur_mem_buffer_t *getBuffer() {
    return std::visit(
        [](auto &&arg) -> ur_mem_buffer_t * {
          if constexpr (std::is_base_of_v<ur_mem_buffer_t,
                                          std::decay_t<decltype(arg)>>) {
            return static_cast<ur_mem_buffer_t *>(&arg);
          } else {
            throw UR_RESULT_ERROR_INVALID_MEM_OBJECT;
          }
        },
        mem);
  }

  ur_mem_image_t *getImage() {
    return std::visit(
        [](auto &&arg) -> ur_mem_image_t * {
          if constexpr (std::is_same_v<ur_mem_image_t,
                                       std::decay_t<decltype(arg)>>) {
            return static_cast<ur_mem_image_t *>(&arg);
          } else {
            throw UR_RESULT_ERROR_INVALID_MEM_OBJECT;
          }
        },
        mem);
  }

  _ur_object *getObject() {
    return std::visit(
        [](auto &&arg) -> _ur_object * {
          return static_cast<_ur_object *>(&arg);
        },
        mem);
  }

  bool isImage() const { return std::holds_alternative<ur_mem_image_t>(mem); }

private:
  template <typename T, typename... Args>
  ur_mem_handle_t_(std::in_place_type_t<T>, Args &&...args)
      : mem(std::in_place_type<T>, std::forward<Args>(args)...) {}

  std::variant<ur_usm_handle_t, ur_integrated_buffer_handle_t,
               ur_discrete_buffer_handle_t, ur_shared_buffer_handle_t,
               ur_mem_sub_buffer_t, ur_mem_image_t>
      mem;
};
