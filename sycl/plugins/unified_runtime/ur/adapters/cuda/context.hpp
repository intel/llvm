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
/// There is no direct mapping between a CUDA context and a UR context,
/// main differences described below:
///
/// <b> CUDA context vs UR context </b>
///
/// One of the main differences between the UR API and the CUDA driver API is
/// that the second modifies the state of the threads by assigning
/// `CUcontext` objects to threads. `CUcontext` objects store data associated
/// with a given device and control access to said device from the user side.
/// UR API context are objects that are passed to functions, and not bound
/// to threads.
/// The _ur_context object doesn't implement this behavior, only holds the
/// CUDA context data. The RAII object \ref ScopedContext implements the active
/// context behavior.
///
/// <b> Primary vs User-defined context </b>
///
/// CUDA has two different types of context, the Primary context,
/// which is usable by all threads on a given process for a given device, and
/// the aforementioned custom contexts.
/// CUDA documentation, and performance analysis, indicates it is recommended
/// to use Primary context whenever possible.
/// Primary context is used as well by the CUDA Runtime API.
/// For UR applications to interop with CUDA Runtime API, they have to use
/// the primary context - and make that active in the thread.
/// The `_ur_context` object can be constructed with a `kind` parameter
/// that allows to construct a Primary or `user-defined` context, so that
/// the UR object interface is always the same.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the PI Context can store a number of callback functions that will be
///  called upon destruction of the UR Context.
///  See proposal for details.
///
struct ur_context_handle_t_ {

  struct deleter_data {
    ur_context_extended_deleter_t function;
    void *user_data;

    void operator()() { function(user_data); }
  };

  using native_type = CUcontext;

  native_type cuContext_;
  ur_device_handle_t deviceId_;
  std::atomic_uint32_t refCount_;

  ur_context_handle_t_(ur_device_handle_t_ *devId)
      : cuContext_{devId->get_context()}, deviceId_{devId}, refCount_{1} {
    urDeviceRetain(deviceId_);
  };

  ~ur_context_handle_t_() { urDeviceRelease(deviceId_); }

  void invoke_extended_deleters() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto &deleter : extended_deleters_) {
      deleter();
    }
  }

  void set_extended_deleter(ur_context_extended_deleter_t function,
                            void *user_data) {
    std::lock_guard<std::mutex> guard(mutex_);
    extended_deleters_.emplace_back(deleter_data{function, user_data});
  }

  ur_device_handle_t get_device() const noexcept { return deviceId_; }

  native_type get() const noexcept { return cuContext_; }

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }

private:
  std::mutex mutex_;
  std::vector<deleter_data> extended_deleters_;
};

namespace {
class ScopedContext {
public:
  ScopedContext(ur_context_handle_t ctxt) {
    if (!ctxt) {
      throw UR_RESULT_ERROR_INVALID_CONTEXT;
    }

    set_context(ctxt->get());
  }

  ScopedContext(CUcontext ctxt) { set_context(ctxt); }

  ~ScopedContext() {}

private:
  void set_context(CUcontext desired) {
    CUcontext original = nullptr;

    UR_CHECK_ERROR(cuCtxGetCurrent(&original));

    // Make sure the desired context is active on the current thread, setting
    // it if necessary
    if (original != desired) {
      UR_CHECK_ERROR(cuCtxSetCurrent(desired));
    }
  }
};
} // namespace
