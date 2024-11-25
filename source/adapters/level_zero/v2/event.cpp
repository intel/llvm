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
#include "queue_api.hpp"

#include "../ur_interface_loader.hpp"

ur_event_handle_t_::ur_event_handle_t_(
    v2::raii::cache_borrowed_event eventAllocation, v2::event_pool *pool)
    : zeEvent(std::move(eventAllocation)), pool(pool),
      adjustedEventStartTimestamp(0), recordEventEndTimestamp(0),
      adjustedEventEndTimestamp(0),
      zeTimerResolution(getDevice()->ZeDeviceProperties->timerResolution),
      timestampMaxValue(getDevice()->getTimestampMask()) {}

void ur_event_handle_t_::resetQueueAndCommand(ur_queue_handle_t hQueue,
                                              ur_command_t commandType) {
  this->hQueue = hQueue;
  this->commandType = commandType;
}

void ur_event_handle_t_::reset() {
  // consider making an abstraction for regular/counter based
  // events if there's more of this type of conditions
  if (!(pool->getFlags() & v2::EVENT_FLAGS_COUNTER)) {
    zeEventHostReset(zeEvent.get());
  }
}

ze_event_handle_t ur_event_handle_t_::getZeEvent() const {
  assert(hQueue);
  assert(commandType != UR_COMMAND_FORCE_UINT32);
  return zeEvent.get();
}

ur_result_t ur_event_handle_t_::retain() {
  RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_event_handle_t_::releaseDeferred() {
  assert(zeEventQueryStatus(zeEvent.get()) == ZE_RESULT_SUCCESS);
  assert(RefCount.load() == 0);

  pool->free(this);
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_event_handle_t_::release() {
  if (!RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  // Need to take a lock before checking if the event is timestamped.
  std::unique_lock<ur_shared_mutex> lock(Mutex);

  if (isTimestamped() && adjustedEventEndTimestamp == 0) {
    // L0 will write end timestamp to this event some time in the future,
    // so we can't release it yet.

    assert(hQueue);
    hQueue->deferEventFree(this);
    return UR_RESULT_SUCCESS;
  }

  pool->free(this);

  return UR_RESULT_SUCCESS;
}

bool ur_event_handle_t_::isTimestamped() const {
  // If we are recording, the start time of the event will be non-zero.
  return adjustedEventStartTimestamp != 0;
}

bool ur_event_handle_t_::isProfilingEnabled() const {
  return pool->getFlags() & v2::EVENT_FLAGS_PROFILING_ENABLED;
}

ur_device_handle_t ur_event_handle_t_::getDevice() const {
  return pool->getProvider()->device();
}

uint64_t ur_event_handle_t_::getEventStartTimestmap() const {
  return adjustedEventStartTimestamp;
}

static uint64_t adjustEndEventTimestamp(uint64_t adjustedStartTimestamp,
                                        uint64_t endTimestamp,
                                        uint64_t timestampMaxValue,
                                        uint64_t timerResolution) {
  // End time needs to be adjusted for resolution and valid bits.
  uint64_t adjustedTimestamp =
      (endTimestamp & timestampMaxValue) * timerResolution;

  // Handle a possible wrap-around (the underlying HW counter is < 64-bit).
  // Note, it will not report correct time if there were multiple wrap
  // arounds, and the longer term plan is to enlarge the capacity of the
  // HW timestamps.
  if (adjustedTimestamp < adjustedStartTimestamp)
    adjustedTimestamp += timestampMaxValue * timerResolution;

  return adjustedTimestamp;
}

uint64_t ur_event_handle_t_::getEventEndTimestamp() {
  // If adjustedEventEndTimestamp on the event is non-zero it means it has
  // collected the result of the queue already. In that case it has been
  // adjusted and is ready for immediate return.
  if (adjustedEventEndTimestamp)
    return adjustedEventEndTimestamp;

  auto status = zeEventQueryStatus(zeEvent.get());
  if (status != ZE_RESULT_SUCCESS) {
    // profiling info not ready
    return 0;
  }

  adjustedEventEndTimestamp =
      adjustEndEventTimestamp(getEventStartTimestmap(), recordEventEndTimestamp,
                              timestampMaxValue, zeTimerResolution);

  return adjustedEventEndTimestamp;
}

void ur_event_handle_t_::recordStartTimestamp() {
  uint64_t deviceStartTimestamp = 0;
  UR_CALL_THROWS(ur::level_zero::urDeviceGetGlobalTimestamps(
      getDevice(), &deviceStartTimestamp, nullptr));

  assert(adjustedEventStartTimestamp == 0);
  adjustedEventStartTimestamp = deviceStartTimestamp;
}

std::pair<uint64_t *, ze_event_handle_t>
ur_event_handle_t_::getEventEndTimestampAndHandle() {
  return {&recordEventEndTimestamp, zeEvent.get()};
}

ur_queue_handle_t ur_event_handle_t_::getQueue() const { return hQueue; }

ur_command_t ur_event_handle_t_::getCommandType() const { return commandType; }

namespace ur::level_zero {
ur_result_t urEventRetain(ur_event_handle_t hEvent) try {
  return hEvent->retain();
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urEventRelease(ur_event_handle_t hEvent) try {
  return hEvent->release();
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urEventWait(uint32_t numEvents,
                        const ur_event_handle_t *phEventWaitList) try {
  for (uint32_t i = 0; i < numEvents; ++i) {
    ZE2UR_CALL(zeEventHostSynchronize,
               (phEventWaitList[i]->getZeEvent(), UINT64_MAX));
  }
  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urEventGetInfo(ur_event_handle_t hEvent, ur_event_info_t propName,
                           size_t propValueSize, void *pPropValue,
                           size_t *pPropValueSizeRet) try {
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
  case UR_EVENT_INFO_COMMAND_QUEUE: {
    return returnValue(ur_queue_handle_t{hEvent->getQueue()});
  }
  case UR_EVENT_INFO_CONTEXT: {
    ur_context_handle_t hContext;
    UR_CALL(::ur::level_zero::urQueueGetInfo(
        hEvent->getQueue(), UR_QUEUE_INFO_CONTEXT, sizeof(hContext),
        reinterpret_cast<void *>(&hContext), nullptr));
    return returnValue(hContext);
  }
  case UR_EVENT_INFO_COMMAND_TYPE: {
    return returnValue(hEvent->getCommandType());
  }
  default:
    logger::error(
        "Unsupported ParamName in urEventGetInfo: ParamName=ParamName={}(0x{})",
        propName, logger::toHex(propName));
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}

ur_result_t urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ///< [in] handle of the event object
    ur_profiling_info_t
        propName, ///< [in] the name of the profiling property to query
    size_t
        propValueSize, ///< [in] size in bytes of the profiling property value
    void *pPropValue,  ///< [out][optional] value of the profiling property
    size_t *pPropValueSizeRet ///< [out][optional] pointer to the actual size in
                              ///< bytes returned in propValue
    ) try {
  std::scoped_lock<ur_shared_mutex> lock(hEvent->Mutex);

  // The event must either have profiling enabled or be recording timestamps.
  bool isTimestampedEvent = hEvent->isTimestamped();
  if (!hEvent->isProfilingEnabled() && !isTimestampedEvent) {
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  UrReturnHelper returnValue(propValueSize, pPropValue, pPropValueSizeRet);

  // For timestamped events we have the timestamps ready directly on the event
  // handle, so we short-circuit the return.
  if (isTimestampedEvent) {
    uint64_t contextStartTime = hEvent->getEventStartTimestmap();
    switch (propName) {
    case UR_PROFILING_INFO_COMMAND_QUEUED:
    case UR_PROFILING_INFO_COMMAND_SUBMIT:
      return returnValue(contextStartTime);
    case UR_PROFILING_INFO_COMMAND_END:
    case UR_PROFILING_INFO_COMMAND_START:
    case UR_PROFILING_INFO_COMMAND_COMPLETE: {
      return returnValue(hEvent->getEventEndTimestamp());
    }
    default:
      logger::error("urEventGetProfilingInfo: not supported ParamName");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }

  ze_kernel_timestamp_result_t tsResult;

  auto zeTimerResolution =
      hEvent->getDevice()->ZeDeviceProperties->timerResolution;
  auto timestampMaxValue = hEvent->getDevice()->getTimestampMask();

  switch (propName) {
  case UR_PROFILING_INFO_COMMAND_START: {
    ZE2UR_CALL(zeEventQueryKernelTimestamp, (hEvent->getZeEvent(), &tsResult));
    uint64_t contextStartTime =
        (tsResult.global.kernelStart & timestampMaxValue) * zeTimerResolution;
    return returnValue(contextStartTime);
  }
  case UR_PROFILING_INFO_COMMAND_END:
  case UR_PROFILING_INFO_COMMAND_COMPLETE: {
    ZE2UR_CALL(zeEventQueryKernelTimestamp, (hEvent->getZeEvent(), &tsResult));

    uint64_t contextStartTime =
        (tsResult.global.kernelStart & timestampMaxValue);

    auto adjustedEndTime =
        adjustEndEventTimestamp(contextStartTime, tsResult.global.kernelEnd,
                                timestampMaxValue, zeTimerResolution);
    return returnValue(adjustedEndTime);
  }
  case UR_PROFILING_INFO_COMMAND_QUEUED:
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
    // Note: No users for this case
    // The "command_submit" time is implemented by recording submission
    // timestamp with a call to urDeviceGetGlobalTimestamps before command
    // enqueue.
    //
    return returnValue(uint64_t{0});
  default:
    logger::error("urEventGetProfilingInfo: not supported ParamName");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  return UR_RESULT_SUCCESS;
} catch (...) {
  return exceptionToResult(std::current_exception());
}
} // namespace ur::level_zero
