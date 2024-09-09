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

ur_result_t urEventWait(uint32_t numEvents,
                        const ur_event_handle_t *phEventWaitList) {
  for (uint32_t i = 0; i < numEvents; ++i) {
    ZE2UR_CALL(zeEventHostSynchronize,
               (phEventWaitList[i]->getZeEvent(), UINT64_MAX));
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urEventGetInfo(ur_event_handle_t hEvent, ur_event_info_t propName,
                           size_t propValueSize, void *pPropValue,
                           size_t *pPropValueSizeRet) {
  UrReturnHelper returnValue(propValueSize, pPropValue, pPropValueSizeRet);

  switch (propName) {
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    auto zeStatus = ZE_CALL_NOCHECK(zeEventQueryStatus, (hEvent->getZeEvent()));

    if (zeStatus == ZE_RESULT_NOT_READY) {
      return returnValue(UR_EVENT_STATUS_SUBMITTED);
    } else {
      return returnValue(UR_EVENT_STATUS_COMPLETE);
    }
  }
  case UR_EVENT_INFO_REFERENCE_COUNT: {
    return returnValue(hEvent->RefCount.load());
  }
  default:
    logger::error(
        "Unsupported ParamName in urEventGetInfo: ParamName=ParamName={}(0x{})",
        propName, logger::toHex(propName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero
