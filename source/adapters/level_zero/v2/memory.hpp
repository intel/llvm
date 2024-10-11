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

  virtual void *getPtr(ur_device_handle_t) = 0;

  inline size_t getSize() { return size; }
  inline ur_context_handle_t getContext() { return hContext; }

protected:
  const ur_context_handle_t hContext;
  const size_t size;
};

struct ur_host_mem_handle_t : public ur_mem_handle_t_ {
  enum class host_ptr_action_t { import, copy };

  ur_host_mem_handle_t(ur_context_handle_t hContext, void *hostPtr, size_t size,
                       host_ptr_action_t useHostPtr);
  ~ur_host_mem_handle_t();

  void *getPtr(ur_device_handle_t) override;

private:
  void *ptr;
};

struct ur_device_mem_handle_t : public ur_mem_handle_t_ {
  ur_device_mem_handle_t(ur_context_handle_t hContext, void *hostPtr,
                         size_t size);
  ~ur_device_mem_handle_t();

  void *getPtr(ur_device_handle_t) override;

private:
  // Vector of per-device allocations indexed by device->Id
  std::vector<void *> deviceAllocations;

  // Specifies device on which the latest allocation resides.
  // If null, there is no allocation.
  ur_device_handle_t activeAllocationDevice;

  ur_result_t migrateBufferTo(ur_device_handle_t hDevice, void *src,
                              size_t size);
};
