//===--------- timeline_semaphore.hpp - Level Zero Adapter ----------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <atomic>
#include <cassert>
#include <map>
#include <utility>

#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "context.hpp"
#include "unified-runtime/ur_api.h"

namespace ur::level_zero::v2 {

// The value of the semaphore lives in one 8-byte host USM allocation that both
// the host and the devices of the context access. A signal writes the value
// into it, and a wait for value N uses a counter-based event whose external
// sync allocation is that memory and whose completion value is N.
//
// Level Zero requires both the completion values of these events and every
// value written to the allocation to stay within the device maximum, so all
// values are checked against the minimum of the maxima of the context devices.
struct ur_exp_timeline_semaphore_handle_t_ : v2::ur_object_t {
  ur_exp_timeline_semaphore_handle_t_(ur_context_handle_t hContext,
                                      uint64_t initialValue, uint64_t maxValue);
  ~ur_exp_timeline_semaphore_handle_t_();

  ur_exp_timeline_semaphore_handle_t_(
      const ur_exp_timeline_semaphore_handle_t_ &) = delete;
  ur_exp_timeline_semaphore_handle_t_ &
  operator=(const ur_exp_timeline_semaphore_handle_t_ &) = delete;

  ur_context_handle_t getContext() const { return hContext.get(); }

  uint64_t *getStorage() const { return reinterpret_cast<uint64_t *>(value); }

  uint64_t getMaxValue() const { return maxValue; }

  // The value never decreases, so a target that is reached stays reached.
  bool isReached(uint64_t target) const {
    return value->load(std::memory_order_acquire) >= target;
  }

  // Returns an event that is complete once the value is at least `target`.
  // The event is owned by the semaphore and stays valid until it is destroyed.
  ze_event_handle_t getWaitEvent(ur_device_handle_t hDevice, uint64_t target);

  void hostSignal(uint64_t newValue) {
    assert(newValue > value->load(std::memory_order_relaxed) &&
           "a timeline semaphore signal must increase the value");
    value->store(newValue, std::memory_order_release);
  }

  ur_result_t release();

  ur::RefCount RefCount;

private:
  v2::raii::ur_context_handle_t hContext;
  std::atomic<uint64_t> *value = nullptr;
  const uint64_t maxValue;

  ur_shared_mutex waitEventsMutex;
  std::map<std::pair<ze_device_handle_t, uint64_t>, raii::ze_event_handle_t>
      waitEvents;
};

} // namespace ur::level_zero::v2
