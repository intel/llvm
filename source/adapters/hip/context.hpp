//===--------- context.hpp - HIP Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <set>

#include "common.hpp"
#include "device.hpp"
#include "platform.hpp"

#include <umf/memory_pool.h>

typedef void (*ur_context_extended_deleter_t)(void *UserData);

///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the UR Context can store a number of callback functions that will be
///  called upon destruction of the UR Context.
///  See proposal for details.
///  https://github.com/codeplaysoftware/standards-proposals/blob/master/extended-context-destruction/index.md
///
///  <b> Memory Management for Devices in a Context <\b>
///
///  A \c ur_mem_handle_t is associated with a \c ur_context_handle_t_, which
///  may refer to multiple devices. Therefore the \c ur_mem_handle_t must
///  handle a native allocation for each device in the context. UR is
///  responsible for automatically handling event dependencies for kernels
///  writing to or reading from the same \c ur_mem_handle_t and migrating memory
///  between native allocations for devices in the same \c ur_context_handle_t_
///  if necessary.
///
struct ur_context_handle_t_ {

  struct deleter_data {
    ur_context_extended_deleter_t Function;
    void *UserData;

    void operator()() { Function(UserData); }
  };

  std::vector<ur_device_handle_t> Devices;

  std::atomic_uint32_t RefCount;

  ur_context_handle_t_(const ur_device_handle_t *Devs, uint32_t NumDevices)
      : Devices{Devs, Devs + NumDevices}, RefCount{1} {};

  ~ur_context_handle_t_() {}

  void invokeExtendedDeleters() {
    std::lock_guard<std::mutex> Guard(Mutex);
    for (auto &Deleter : ExtendedDeleters) {
      Deleter();
    }
  }

  void setExtendedDeleter(ur_context_extended_deleter_t Function,
                          void *UserData) {
    std::lock_guard<std::mutex> Guard(Mutex);
    ExtendedDeleters.emplace_back(deleter_data{Function, UserData});
  }

  const std::vector<ur_device_handle_t> &getDevices() const noexcept {
    return Devices;
  }

  // Gets the index of the device relative to other devices in the context
  size_t getDeviceIndex(ur_device_handle_t hDevice) {
    auto It = std::find(Devices.begin(), Devices.end(), hDevice);
    assert(It != Devices.end());
    return std::distance(Devices.begin(), It);
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  void addPool(ur_usm_pool_handle_t Pool);

  void removePool(ur_usm_pool_handle_t Pool);

  ur_usm_pool_handle_t getOwningURPool(umf_memory_pool_t *UMFPool);

private:
  std::mutex Mutex;
  std::vector<deleter_data> ExtendedDeleters;
  std::set<ur_usm_pool_handle_t> PoolHandles;
};
