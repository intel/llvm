//===--------- event.cpp - CUDA Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "event.hpp"
#include "context.hpp"
#include "device.hpp"
#include "queue.hpp"
#include "ur_api.h"
#include "ur_util.hpp"

#include <cassert>
#include <cuda.h>

ur_event_handle_t_::ur_event_handle_t_(ur_command_t Type,
                                       ur_queue_handle_t Queue, CUstream Stream,
                                       uint32_t StreamToken)
    : handle_base(), CommandType{Type}, StreamToken{StreamToken}, Queue{Queue},
      Stream{Stream}, Context{Queue->getContext()} {
  auto flag = CU_EVENT_DISABLE_TIMING;

  // If profiling information is required
  if (Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE ||
      Type == UR_COMMAND_TIMESTAMP_RECORDING_EXP) {
    HasProfiling = true;
    flag = CU_EVENT_DEFAULT;
    UR_CHECK_ERROR(cuEventCreate(&EvQueued, flag));
    UR_CHECK_ERROR(cuEventCreate(&EvStart, flag));
  }

  UR_CHECK_ERROR(cuEventCreate(&EvEnd, flag));

  urQueueRetain(Queue);
  urContextRetain(Context);
}

ur_event_handle_t_::ur_event_handle_t_(ur_context_handle_t Context,
                                       CUevent EventNative)
    : handle_base(), CommandType{UR_COMMAND_EVENTS_WAIT}, HasProfiling{false},
      IsInterop{true}, StreamToken{std::numeric_limits<uint32_t>::max()},
      EvEnd{EventNative}, EvStart{nullptr}, EvQueued{nullptr}, Queue{nullptr},
      Stream{nullptr}, Context{Context} {
  urContextRetain(Context);
}

ur_event_handle_t_::~ur_event_handle_t_() {
  if (Queue != nullptr) {
    urQueueRelease(Queue);
  }
  urContextRelease(Context);
}

ur_result_t ur_event_handle_t_::release() {
  if (!HasOwnership)
    return UR_RESULT_SUCCESS;

  assert(Queue != nullptr);

  UR_CHECK_ERROR(cuEventDestroy(EvEnd));

  if (HasProfiling) {
    UR_CHECK_ERROR(cuEventDestroy(EvQueued));
    UR_CHECK_ERROR(cuEventDestroy(EvStart));
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_event_handle_t_::start() {
  assert(!isStarted());
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    if (HasProfiling) {
      UR_CHECK_ERROR(cuEventRecord(EvQueued, Queue->getHostSubmitTimeStream()));
      UR_CHECK_ERROR(cuEventRecord(EvStart, Stream));
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }

  IsStarted = true;
  return Result;
}

ur_result_t ur_event_handle_t_::record() {
  if (isRecorded() || !isStarted()) {
    return UR_RESULT_ERROR_INVALID_EVENT;
  }

  UR_ASSERT(Queue, UR_RESULT_ERROR_INVALID_QUEUE);

  try {
    EventID = Queue->getNextEventId();
    if (EventID == 0) {
      die("Unrecoverable program state reached in event identifier overflow");
    }
    UR_CHECK_ERROR(cuEventRecord(EvEnd, Stream));
  } catch (ur_result_t error) {
    return error;
  }

  IsRecorded = true;
  return UR_RESULT_SUCCESS;
}

ur_result_t ur_event_handle_t_::wait() {
  try {
    UR_CHECK_ERROR(cuEventSynchronize(EvEnd));
    HasBeenWaitedOn = true;
  } catch (ur_result_t error) {
    return error;
  }

  return UR_RESULT_SUCCESS;
}

bool ur_event_handle_t_::isCompleted() const noexcept try {
  if (!IsRecorded) {
    return false;
  }
  if (!HasBeenWaitedOn) {
    const CUresult Result = cuEventQuery(EvEnd);
    if (Result != CUDA_SUCCESS && Result != CUDA_ERROR_NOT_READY) {
      UR_CHECK_ERROR(Result);
      return false;
    }
    if (Result == CUDA_ERROR_NOT_READY) {
      return false;
    }
  }
  return true;
} catch (...) {
  return exceptionToResult(std::current_exception()) == UR_RESULT_SUCCESS;
}

uint64_t ur_event_handle_t_::getQueuedTime() const {
  assert(isStarted());
  return Queue->getDevice()->getElapsedTime(EvQueued);
}

uint64_t ur_event_handle_t_::getStartTime() const {
  assert(isStarted());
  return Queue->getDevice()->getElapsedTime(EvStart);
}

uint64_t ur_event_handle_t_::getEndTime() const {
  assert(isStarted() && isRecorded());
  return Queue->getDevice()->getElapsedTime(EvEnd);
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropValueSizeRet) {
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  switch (propName) {
  case UR_EVENT_INFO_COMMAND_QUEUE: {
    // If the runtime owns the native handle, we have reference to the queue.
    // Otherwise, the event handle comes from an interop API with no RT refs.
    if (!hEvent->getQueue()) {
      setErrorMessage("Command queue info cannot be queried for the event. The "
                      "event object was created from a native event and has no "
                      "valid reference to a command queue.",
                      UR_RESULT_ERROR_INVALID_VALUE);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }
    return ReturnValue(hEvent->getQueue());
  }
  case UR_EVENT_INFO_COMMAND_TYPE:
    return ReturnValue(hEvent->getCommandType());
  case UR_EVENT_INFO_REFERENCE_COUNT:
    return ReturnValue(hEvent->getReferenceCount());
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS:
    return ReturnValue(hEvent->getExecutionStatus());
  case UR_EVENT_INFO_CONTEXT:
    return ReturnValue(hEvent->getContext());
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

/// Obtain profiling information from PI CUDA events
/// \TODO Timings from CUDA are only elapsed time.
UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ur_profiling_info_t propName,
    size_t propValueSize, void *pPropValue, size_t *pPropValueSizeRet) {
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  ur_queue_handle_t Queue = hEvent->getQueue();
  if (Queue == nullptr || !hEvent->hasProfiling()) {
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  switch (propName) {
  case UR_PROFILING_INFO_COMMAND_QUEUED:
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
    // Note: No user for this case
    return ReturnValue(static_cast<uint64_t>(hEvent->getQueuedTime()));
  case UR_PROFILING_INFO_COMMAND_START:
    return ReturnValue(static_cast<uint64_t>(hEvent->getStartTime()));
  case UR_PROFILING_INFO_COMMAND_END:
    return ReturnValue(static_cast<uint64_t>(hEvent->getEndTime()));
  case UR_PROFILING_INFO_COMMAND_COMPLETE:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default:
    break;
  }
  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventSetCallback(ur_event_handle_t,
                                                       ur_execution_info_t,
                                                       ur_event_callback_t,
                                                       void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  try {
    // Interop events don't have an associated queue, so get device through
    // context
    ScopedContext Active(phEventWaitList[0]->getContext()->getDevices()[0]);

    auto WaitFunc = [](ur_event_handle_t Event) -> ur_result_t {
      UR_ASSERT(Event, UR_RESULT_ERROR_INVALID_EVENT);

      return Event->wait();
    };
    return forLatestEvents(phEventWaitList, numEvents, WaitFunc);
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(ur_event_handle_t hEvent) {
  const auto RefCount = hEvent->incrementReferenceCount();

  if (RefCount == 0) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(ur_event_handle_t hEvent) {
  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  if (hEvent->getReferenceCount() == 0) {
    return UR_RESULT_ERROR_INVALID_EVENT;
  }

  // decrement ref count. If it is 0, delete the event.
  if (hEvent->decrementReferenceCount() == 0) {
    std::unique_ptr<ur_event_handle_t_> event_ptr{hEvent};
    ur_result_t Result = UR_RESULT_ERROR_INVALID_EVENT;
    try {
      Result = hEvent->release();
    } catch (...) {
      Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }
    return Result;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ur_native_handle_t *phNativeEvent) {
  *phNativeEvent = reinterpret_cast<ur_native_handle_t>(hEvent->get());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ur_context_handle_t hContext,
    const ur_event_native_properties_t * /*pProperties*/,
    ur_event_handle_t *phEvent) {

  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    EventPtr = std::make_unique<ur_event_handle_t_>(
        hContext, reinterpret_cast<CUevent>(hNativeEvent));
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  *phEvent = EventPtr.release();

  return UR_RESULT_SUCCESS;
}
