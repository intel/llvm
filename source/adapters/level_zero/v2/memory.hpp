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

struct ur_mem_handle_t_ : _ur_object {
  ur_mem_handle_t_(ur_context_handle_t hContext, size_t size);
  virtual ~ur_mem_handle_t_() = default;

  enum class access_mode_t {
    read_write,
    read_only,
    write_only,
    write_invalidate
  };

  // Following functions should always be called under the lock.
  virtual void *
  getDevicePtr(ur_device_handle_t, access_mode_t, size_t offset, size_t size,
               std::function<void(void *src, void *dst, size_t)> mecmpy) = 0;
  virtual void *
  mapHostPtr(access_mode_t, size_t offset, size_t size,
             std::function<void(void *src, void *dst, size_t)> memcpy) = 0;
  virtual void
  unmapHostPtr(void *pMappedPtr,
               std::function<void(void *src, void *dst, size_t)> memcpy) = 0;

  inline size_t getSize() { return size; }
  inline ur_context_handle_t getContext() { return hContext; }

protected:
  const ur_context_handle_t hContext;
  const size_t size;
};

struct ur_usm_handle_t_ : ur_mem_handle_t_ {
  ur_usm_handle_t_(ur_context_handle_t hContext, size_t size, const void *ptr);
  ~ur_usm_handle_t_();

  void *
  getDevicePtr(ur_device_handle_t, access_mode_t, size_t offset, size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(access_mode_t, size_t offset, size_t size,
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
                             size_t size, host_ptr_action_t useHostPtr);
  ~ur_integrated_mem_handle_t();

  void *
  getDevicePtr(ur_device_handle_t, access_mode_t, size_t offset, size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(access_mode_t, size_t offset, size_t size,
                   std::function<void(void *src, void *dst, size_t)>) override;
  void unmapHostPtr(void *pMappedPtr,
                    std::function<void(void *src, void *dst, size_t)>) override;

private:
  void *ptr;
};

struct host_allocation_desc_t {
  host_allocation_desc_t(void *ptr, size_t size, size_t offset,
                         ur_mem_handle_t_::access_mode_t access)
      : ptr(ptr), size(size), offset(offset), access(access) {}

  void *ptr;
  size_t size;
  size_t offset;
  ur_mem_handle_t_::access_mode_t access;
};

// Manages memory buffer for discrete GPU.
// Memory is allocated on the device and migrated/copies if necessary.
struct ur_discrete_mem_handle_t : public ur_mem_handle_t_ {
  ur_discrete_mem_handle_t(ur_context_handle_t hContext, void *hostPtr,
                           size_t size);
  ~ur_discrete_mem_handle_t();

  void *
  getDevicePtr(ur_device_handle_t, access_mode_t, size_t offset, size_t size,
               std::function<void(void *src, void *dst, size_t)>) override;
  void *mapHostPtr(access_mode_t, size_t offset, size_t size,
                   std::function<void(void *src, void *dst, size_t)>) override;
  void unmapHostPtr(void *pMappedPtr,
                    std::function<void(void *src, void *dst, size_t)>) override;

private:
  // Vector of per-device allocations indexed by device->Id
  std::vector<void *> deviceAllocations;

  // Specifies device on which the latest allocation resides.
  // If null, there is no allocation.
  ur_device_handle_t activeAllocationDevice;

  std::vector<host_allocation_desc_t> hostAllocations;

  ur_result_t migrateBufferTo(ur_device_handle_t hDevice, void *src,
                              size_t size);
};
