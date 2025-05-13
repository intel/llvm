//===--------- context.hpp - CUDA Adapter ---------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <memory>
#include <ur_api.h>

#include <atomic>
#include <mutex>
#include <set>
#include <vector>

#include "adapter.hpp"
#include "common.hpp"
#include "device.hpp"
#include "umf_helpers.hpp"

#include <umf/memory_pool.h>

typedef void (*ur_context_extended_deleter_t)(void *user_data);

/// UR context mapping to a CUDA context object.
///
/// There is no direct mapping between a CUDA context and a UR context.
/// The main differences are described below:
///
/// <b> CUDA context vs UR context </b>
///
/// One of the main differences between the UR API and the CUDA driver API is
/// that the second modifies the state of the threads by assigning
/// \c CUcontext objects to threads. \c CUcontext objects store data associated
/// with a given device and control access to said device from the user side.
/// UR API context are objects that are passed to functions, and not bound
/// to threads.
///
/// Since the \c ur_context_handle_t can contain multiple devices, and a \c
/// CUcontext refers to only a single device, the \c CUcontext is more tightly
/// coupled to a \c ur_device_handle_t than a \c ur_context_handle_t. In order
/// to remove some ambiguities about the different semantics of \c
/// \c ur_context_handle_t and native \c CUcontext, we access the native \c
/// CUcontext solely through the \c ur_device_handle_t class, by using the
/// object \ref ScopedContext, which sets the active device (by setting the
/// active native \c CUcontext).
///
/// <b> Primary vs User-defined \c CUcontext </b>
///
/// CUDA has two different types of \c CUcontext, the Primary context, which is
/// usable by all threads on a given process for a given device, and the
/// aforementioned custom \c CUcontext s. The CUDA documentation, confirmed with
/// performance analysis, suggest using the Primary context whenever possible.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the PI Context can store a number of callback functions that will be
///  called upon destruction of the UR Context.
///  See proposal for details.
///  https://github.com/codeplaysoftware/standards-proposals/blob/master/extended-context-destruction/index.md
///
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
///

struct ur_context_handle_t_ {

  struct deleter_data {
    ur_context_extended_deleter_t Function;
    void *UserData;

    void operator()() { Function(UserData); }
  };

  std::vector<ur_device_handle_t> Devices;
  std::atomic_uint32_t RefCount;

  // UMF CUDA memory provider and pool for the host memory
  // (UMF_MEMORY_TYPE_HOST)
  umf_memory_provider_handle_t MemoryProviderHost = nullptr;
  umf_memory_pool_handle_t MemoryPoolHost = nullptr;

  ur_context_handle_t_(const ur_device_handle_t *Devs, uint32_t NumDevices)
      : Devices{Devs, Devs + NumDevices}, RefCount{1} {
    // Create UMF CUDA memory provider for the host memory
    // (UMF_MEMORY_TYPE_HOST) from any device (Devices[0] is used here, because
    // it is guaranteed to exist).
    UR_CHECK_ERROR(umf::CreateProviderPool(
        0, Devices[0]->getNativeContext(), UMF_MEMORY_TYPE_HOST,
        &MemoryProviderHost, &MemoryPoolHost));
    UR_CHECK_ERROR(urAdapterRetain(ur::cuda::adapter));
  };

  ~ur_context_handle_t_() {
    if (MemoryPoolHost) {
      umfPoolDestroy(MemoryPoolHost);
    }
    if (MemoryProviderHost) {
      umfMemoryProviderDestroy(MemoryProviderHost);
    }
    UR_CHECK_ERROR(urAdapterRelease(ur::cuda::adapter));
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

namespace {
class ScopedContext {
public:
  ScopedContext(ur_device_handle_t Device) {
    if (!Device) {
      throw UR_RESULT_ERROR_INVALID_DEVICE;
    }
    setContext(Device->getNativeContext());
  }

  ScopedContext(CUcontext NativeContext) { setContext(NativeContext); }

  ~ScopedContext() {}

private:
  void setContext(CUcontext Desired) {
    CUcontext Original = nullptr;

    UR_CHECK_ERROR(cuCtxGetCurrent(&Original));

    // Make sure the desired context is active on the current thread, setting
    // it if necessary
    if (Original != Desired) {
      UR_CHECK_ERROR(cuCtxSetCurrent(Desired));
    }
  }
};
} // namespace
