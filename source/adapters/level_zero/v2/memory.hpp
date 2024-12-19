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

#include <ur_api.h>

#include "../device.hpp"
#include "common.hpp"

using usm_unique_ptr_t = std::unique_ptr<void, std::function<void(void *)>>;

struct ur_mem_handle_t_ : private _ur_object {
  enum class device_access_mode_t { read_write, read_only, write_only };

  ur_mem_handle_t_(ur_context_handle_t hContext, size_t size,
                   device_access_mode_t accesMode);
  virtual ~ur_mem_handle_t_() = default;

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

  inline device_access_mode_t getDeviceAccessMode() const { return accessMode; }
  inline ur_context_handle_t getContext() const { return hContext; }
  inline ReferenceCounter &getRefCount() { return RefCount; }

  virtual size_t getSize() const;
  virtual ur_shared_mutex &getMutex();

protected:
  const device_access_mode_t accessMode;
  const ur_context_handle_t hContext;
  const size_t size;
};

// non-owning buffer wrapper around USM pointer
struct ur_usm_handle_t_ : ur_mem_handle_t_ {
  ur_usm_handle_t_(ur_context_handle_t hContext, size_t size, const void *ptr);
  ~ur_usm_handle_t_();

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
struct ur_integrated_mem_handle_t : public ur_mem_handle_t_ {
  enum class host_ptr_action_t { import, copy };

  ur_integrated_mem_handle_t(ur_context_handle_t hContext, void *hostPtr,
                             size_t size, host_ptr_action_t useHostPtr,
                             device_access_mode_t accesMode);

  ur_integrated_mem_handle_t(ur_context_handle_t hContext, void *hostPtr,
                             size_t size, device_access_mode_t accesMode,
                             bool ownHostPtr);

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
struct ur_discrete_mem_handle_t : public ur_mem_handle_t_ {
  // If hostPtr is not null, the buffer is allocated immediately on the
  // first device in the context. Otherwise, the buffer is allocated on
  // firt getDevicePtr call.
  ur_discrete_mem_handle_t(ur_context_handle_t hContext, void *hostPtr,
                           size_t size, device_access_mode_t accesMode);
  ~ur_discrete_mem_handle_t();

  // Create buffer on top of existing device memory.
  ur_discrete_mem_handle_t(ur_context_handle_t hContext,
                           ur_device_handle_t hDevice, void *devicePtr,
                           size_t size, device_access_mode_t accesMode,
                           void *writeBackMemory, bool ownDevicePtr);

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

struct ur_mem_sub_buffer_t : public ur_mem_handle_t_ {
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

  size_t getSize() const override;
  ur_shared_mutex &getMutex() override;

private:
  ur_mem_handle_t hParent;
  size_t offset;
  size_t size;
};
