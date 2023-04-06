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

// We need this declaration temporarily while UR and PI share ScopedContext
class _pi_context;
using pi_context = _pi_context *;

#include "common.hpp"
#include "device.hpp"

typedef void (*ur_context_extended_deleter_t)(void *user_data);

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
  // TODO(ur): Needed for compatibility with PI; once the CUDA PI plugin is
  // fully moved over we can drop this constructor
  ScopedContext(pi_context ctxt);

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
