//===--------- context.hpp - HIP Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "device.hpp"
#include "platform.hpp"

typedef void (*ur_context_extended_deleter_t)(void *user_data);

/// UR context mapping to a HIP context object.
///
/// There is no direct mapping between a HIP context and a UR context,
/// main differences described below:
///
/// <b> HIP context vs UR context </b>
///
/// One of the main differences between the UR API and the HIP driver API is
/// that the second modifies the state of the threads by assigning
/// `hipCtx_t` objects to threads. `hipCtx_t` objects store data associated
/// with a given device and control access to said device from the user side.
/// UR API context are objects that are passed to functions, and not bound
/// to threads.
/// The ur_context_handle_t_ object doesn't implement this behavior, only holds the
/// HIP context data. The RAII object \ref ScopedContext implements the active
/// context behavior.
///
/// <b> Primary vs User-defined context </b>
///
/// HIP has two different types of context, the Primary context,
/// which is usable by all threads on a given process for a given device, and
/// the aforementioned custom contexts.
/// HIP documentation, and performance analysis, indicates it is recommended
/// to use Primary context whenever possible.
/// Primary context is used as well by the HIP Runtime API.
/// For UR applications to interop with HIP Runtime API, they have to use
/// the primary context - and make that active in the thread.
/// The `ur_context_handle_t_` object can be constructed with a `kind` parameter
/// that allows to construct a Primary or `user-defined` context, so that
/// the UR object interface is always the same.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the UR Context can store a number of callback functions that will be
///  called upon destruction of the UR Context.
///  See proposal for details.
///
struct ur_context_handle_t_ {

  struct deleter_data {
    ur_context_extended_deleter_t function;
    void *user_data;

    void operator()() { function(user_data); }
  };

  using native_type = hipCtx_t;

  enum class kind { primary, user_defined } kind_;
  native_type hipContext_;
  ur_device_handle_t deviceId_;
  std::atomic_uint32_t refCount_;

  ur_context_handle_t_(kind k, hipCtx_t ctxt, ur_device_handle_t devId)
      : kind_{k}, hipContext_{ctxt}, deviceId_{devId}, refCount_{1} {
    deviceId_->set_context(this);
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

  native_type get() const noexcept { return hipContext_; }

  bool is_primary() const noexcept { return kind_ == kind::primary; }

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }

private:
  std::mutex mutex_;
  std::vector<deleter_data> extended_deleters_;
};

namespace {
/// RAII type to guarantee recovering original HIP context
/// Scoped context is used across all UR HIP plugin implementation
/// to activate the UR Context on the current thread, matching the
/// HIP driver semantics where the context used for the HIP Driver
/// API is the one active on the thread.
/// The implementation tries to avoid replacing the hipCtx_t if it cans
class ScopedContext {
  ur_context_handle_t placedContext_;
  hipCtx_t original_;
  bool needToRecover_;

public:
  ScopedContext(ur_context_handle_t ctxt)
      : placedContext_{ctxt}, needToRecover_{false} {

    if (!placedContext_) {
      throw UR_RESULT_ERROR_INVALID_CONTEXT;
    }

    hipCtx_t desired = placedContext_->get();
    UR_CHECK_ERROR(hipCtxGetCurrent(&original_));
    if (original_ != desired) {
      // Sets the desired context as the active one for the thread
      UR_CHECK_ERROR(hipCtxSetCurrent(desired));
      if (original_ == nullptr) {
        // No context is installed on the current thread
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the UR context referring to the
        // same underlying HIP context are destroyed. This emulates
        // the behaviour of the HIP runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
      } else {
        needToRecover_ = true;
      }
    }
  }

  ~ScopedContext() {
    if (needToRecover_) {
      UR_CHECK_ERROR(hipCtxSetCurrent(original_));
    }
  }
};
} // namespace
