//===--------- event.cpp - Level Zero Adapter -----------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ze_api.h>

#include "event.hpp"
#include "event_pool.hpp"
#include "event_provider.hpp"

ur_event_handle_t_::ur_event_handle_t_(v2::event_allocation eventAllocation,
                                       v2::event_pool *pool)
    : type(eventAllocation.type), zeEvent(std::move(eventAllocation.borrow)),
      pool(pool) {}

void ur_event_handle_t_::reset() {
  // consider make an abstraction for regular/counter based
  // events if there's more of this type of conditions
  if (type == v2::event_type::EVENT_REGULAR) {
    zeEventHostReset(zeEvent.get());
  }
}

ze_event_handle_t ur_event_handle_t_::getZeEvent() const {
  return zeEvent.get();
}

ur_result_t ur_event_handle_t_::retain() {
  RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_event_handle_t_::release() {
  if (!RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  pool->free(this);

  return UR_RESULT_SUCCESS;
}

namespace ur::level_zero {
ur_result_t urEventRetain(ur_event_handle_t hEvent) { return hEvent->retain(); }

ur_result_t urEventRelease(ur_event_handle_t hEvent) {
  return hEvent->release();
}
} // namespace ur::level_zero
