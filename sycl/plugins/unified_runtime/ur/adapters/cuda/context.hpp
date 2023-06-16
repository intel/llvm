//===--------- context.hpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <ur_api.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "common.hpp"
#include "device.hpp"

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
/// `CUcontext` objects to threads. `CUcontext` objects store data associated
/// with a given device and control access to said device from the user side.
/// UR API context are objects that are passed to functions, and not bound
/// to threads.
/// The ur_context_handle_t_ object doesn't implement this behavior. It only
/// holds the CUDA context data. The RAII object \ref ScopedContext implements
/// the active context behavior.
///
/// <b> Primary vs User-defined context </b>
///
/// CUDA has two different types of context, the Primary context,
/// which is usable by all threads on a given process for a given device, and
/// the aforementioned custom contexts.
/// The CUDA documentation, confirmed with performance analysis, suggest using
/// the Primary context whenever possible.
/// The Primary context is also used by the CUDA Runtime API.
/// For UR applications to interop with CUDA Runtime API, they have to use
/// the primary context - and make that active in the thread.
/// The `ur_context_handle_t_` object can be constructed with a `kind` parameter
/// that allows to construct a Primary or `user-defined` context, so that
/// the UR object interface is always the same.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the PI Context can store a number of callback functions that will be
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

  using native_type = CUcontext;

  native_type CUContext;
  ur_device_handle_t DeviceID;
  std::atomic_uint32_t RefCount;

  ur_context_handle_t_(ur_device_handle_t_ *DevID)
      : CUContext{DevID->getContext()}, DeviceID{DevID}, RefCount{1} {
    urDeviceRetain(DeviceID);
  };

  ~ur_context_handle_t_() { urDeviceRelease(DeviceID); }

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

  ur_device_handle_t getDevice() const noexcept { return DeviceID; }

  native_type get() const noexcept { return CUContext; }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

private:
  std::mutex Mutex;
  std::vector<deleter_data> ExtendedDeleters;
};

namespace {
class ScopedContext {
public:
  ScopedContext(ur_context_handle_t Context) {
    if (!Context) {
      throw UR_RESULT_ERROR_INVALID_CONTEXT;
    }

    setContext(Context->get());
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
