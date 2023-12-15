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

/// UR context mapping to a HIP context object.
///
/// There is no direct mapping between a HIP context and a UR context.
/// The main differences are described below:
///
/// <b> HIP context vs UR context </b>
///
/// One of the main differences between the UR API and the HIP driver API is
/// that the second modifies the state of the threads by assigning
/// \c hipCtx_t objects to threads. \c hipCtx_t objects store data associated
/// with a given device and control access to said device from the user side.
/// UR API context are objects that are passed to functions, and not bound
/// to threads.
///
/// Since the \c ur_context_handle_t can contain multiple devices, and a \c
/// hipCtx_t refers to only a single device, the \c hipCtx_t is more tightly
/// coupled to a \c ur_device_handle_t than a \c ur_context_handle_t. In order
/// to remove some ambiguities about the different semantics of \c
/// \c ur_context_handle_t and native \c hipCtx_t, we access the native \c
/// hipCtx_t solely through the \c ur_device_handle_t class, by using the object
/// \ref ScopedContext, which sets the active device (by setting the active
/// native \c hipCtx_t).
///
/// <b> Primary vs User-defined \c hipCtx_t </b>
///
/// HIP has two different types of \c hipCtx_t, the Primary context, which is
/// usable by all threads on a given process for a given device, and the
/// aforementioned custom \c hipCtx_t s. The HIP documentation, confirmed with
/// performance analysis, suggest using the Primary context whenever possible.
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

  using native_type = hipCtx_t;

  std::vector<ur_device_handle_t> Devices;

  std::atomic_uint32_t RefCount;

  ur_context_handle_t_(const ur_device_handle_t *Devs, uint32_t NumDevices)
      : Devices{Devs, Devs + NumDevices}, RefCount{1} {
    for (auto &Dev : Devices) {
      urDeviceRetain(Dev);
    }
  };

  ~ur_context_handle_t_() {
    for (auto &Dev : Devices) {
      urDeviceRelease(Dev);
    }
  }

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

namespace {
/// Scoped context is used across all UR HIP plugin implementation to activate
/// the native Context on the current thread. The ScopedContext does not
/// reinstate the previous context as all operations in the hip adapter that
/// require an active context, set the active context and don't rely on context
/// reinstation
class ScopedContext {
public:
  ScopedContext(ur_device_handle_t hDevice) {
    hipCtx_t Original{};

    if (!hDevice) {
      throw UR_RESULT_ERROR_INVALID_DEVICE;
    }

    hipCtx_t Desired = hDevice->getNativeContext();
    UR_CHECK_ERROR(hipCtxGetCurrent(&Original));
    if (Original != Desired) {
      // Sets the desired context as the active one for the thread
      UR_CHECK_ERROR(hipCtxSetCurrent(Desired));
    }
  }
};
} // namespace
