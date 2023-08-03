//===--------- context.hpp - HIP Adapter ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <set>
#include <unordered_map>

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
/// `hipCtx_t` objects to threads. `hipCtx_t` objects store data associated
/// with a given device and control access to said device from the user side.
/// UR API context are objects that are passed to functions, and not bound
/// to threads.
/// The ur_context_handle_t_ object doesn't implement this behavior. It only
/// holds the HIP context data. The RAII object \ref ScopedDevice implements
/// the active context behavior.
///
/// <b> Primary vs UserDefined context </b>
///
/// HIP has two different types of context, the Primary context,
/// which is usable by all threads on a given process for a given device, and
/// the aforementioned custom contexts.
/// The HIP documentation, and performance analysis, suggest using the Primary
/// context whenever possible. The Primary context is also used by the HIP
/// Runtime API. For UR applications to interop with HIP Runtime API, they have
/// to use the primary context - and make that active in the thread. The
/// `ur_context_handle_t_` object can be constructed with a `kind` parameter
/// that allows to construct a Primary or `UserDefined` context, so that
/// the UR object interface is always the same.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the UR Context can store a number of callback functions that will be
///  called upon destruction of the UR Context.
///  See proposal for details.
///  https://github.com/codeplaysoftware/standards-proposals/blob/master/extended-context-destruction/index.md
///
struct ur_context_handle_t_ {

  struct deleter_data {
    ur_context_extended_deleter_t Function;
    void *UserData;

    void operator()() { Function(UserData); }
  };

  using native_type = hipCtx_t;

  ur_device_handle_t DeviceId;
  std::atomic_uint32_t RefCount;

  ur_context_handle_t_(ur_device_handle_t DevId)
      : DeviceId{DevId}, RefCount{1} {
    urDeviceRetain(DeviceId);
  };

  ~ur_context_handle_t_() { urDeviceRelease(DeviceId); }

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

  ur_device_handle_t getDevice() const noexcept { return DeviceId; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  void addPool(ur_usm_pool_handle_t Pool);

  void removePool(ur_usm_pool_handle_t Pool);

  ur_usm_pool_handle_t getOwningURPool(umf_memory_pool_t *UMFPool);

  /// We need to keep track of USM mappings in AMD HIP, as certain extra
  /// synchronization *is* actually required for correctness.
  /// During kernel enqueue we must dispatch a prefetch for each kernel argument
  /// that points to a USM mapping to ensure the mapping is correctly
  /// populated on the device (https://github.com/intel/llvm/issues/7252). Thus,
  /// we keep track of mappings in the context, and then check against them just
  /// before the kernel is launched. The stream against which the kernel is
  /// launched is not known until enqueue time, but the USM mappings can happen
  /// at any time. Thus, they are tracked on the context used for the urUSM*
  /// mapping.
  ///
  /// The three utility function are simple wrappers around a mapping from a
  /// pointer to a size.
  void addUSMMapping(void *Ptr, size_t Size) {
    std::lock_guard<std::mutex> Guard(Mutex);
    assert(USMMappings.find(Ptr) == USMMappings.end() &&
           "mapping already exists");
    USMMappings[Ptr] = Size;
  }

  void removeUSMMapping(const void *Ptr) {
    std::lock_guard<std::mutex> guard(Mutex);
    auto It = USMMappings.find(Ptr);
    if (It != USMMappings.end())
      USMMappings.erase(It);
  }

  std::pair<const void *, size_t> getUSMMapping(const void *Ptr) {
    std::lock_guard<std::mutex> Guard(Mutex);
    auto It = USMMappings.find(Ptr);
    // The simple case is the fast case...
    if (It != USMMappings.end())
      return *It;

    // ... but in the failure case we have to fall back to a full scan to search
    // for "offset" pointers in case the user passes in the middle of an
    // allocation. We have to do some not-so-ordained-by-the-standard ordered
    // comparisons of pointers here, but it'll work on all platforms we support.
    uintptr_t PtrVal = (uintptr_t)Ptr;
    for (std::pair<const void *, size_t> Pair : USMMappings) {
      uintptr_t BaseAddr = (uintptr_t)Pair.first;
      uintptr_t EndAddr = BaseAddr + Pair.second;
      if (PtrVal > BaseAddr && PtrVal < EndAddr) {
        // If we've found something now, offset *must* be nonzero
        assert(Pair.second);
        return Pair;
      }
    }
    return {nullptr, 0};
  }

private:
  std::mutex Mutex;
  std::vector<deleter_data> ExtendedDeleters;
  std::unordered_map<const void *, size_t> USMMappings;
  std::set<ur_usm_pool_handle_t> PoolHandles;
};
