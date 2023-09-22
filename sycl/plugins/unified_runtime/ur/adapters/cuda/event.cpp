//===--------- event.cpp - CUDA Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "event.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "queue.hpp"

#include <cassert>
#include <cuda.h>

ur_event_handle_t_::ur_event_handle_t_(ur_command_t Type,
                                       ur_context_handle_t Context,
                                       ur_queue_handle_t Queue, CUstream Stream,
                                       uint32_t StreamToken)
    : CommandType{Type}, RefCount{1}, HasOwnership{true},
      HasBeenWaitedOn{false}, IsRecorded{false}, IsStarted{false},
      StreamToken{StreamToken}, EvEnd{nullptr}, EvStart{nullptr},
      EvQueued{nullptr}, Queue{Queue}, Stream{Stream}, Context{Context} {

  bool ProfilingEnabled = Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE;

  UR_CHECK_ERROR(cuEventCreate(
      &EvEnd, ProfilingEnabled ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING));

  if (ProfilingEnabled) {
    UR_CHECK_ERROR(cuEventCreate(&EvQueued, CU_EVENT_DEFAULT));
    UR_CHECK_ERROR(cuEventCreate(&EvStart, CU_EVENT_DEFAULT));
  }

  if (Queue != nullptr) {
    urQueueRetain(Queue);
  }
  urContextRetain(Context);
}

ur_event_handle_t_::ur_event_handle_t_(ur_context_handle_t Context,
                                       CUevent EventNative)
    : CommandType{UR_COMMAND_EVENTS_WAIT}, RefCount{1}, HasOwnership{false},
      HasBeenWaitedOn{false}, IsRecorded{false}, IsStarted{false},
      StreamToken{std::numeric_limits<uint32_t>::max()}, EvEnd{EventNative},
      EvStart{nullptr}, EvQueued{nullptr}, Queue{nullptr}, Context{Context} {
  urContextRetain(Context);
}

ur_event_handle_t_::~ur_event_handle_t_() {
  if (Queue != nullptr) {
    urQueueRelease(Queue);
  }
  urContextRelease(Context);
}

ur_result_t ur_event_handle_t_::start() {
  assert(!isStarted());
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    if (Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE) {
      // NOTE: This relies on the default stream to be unused.
      UR_CHECK_ERROR(cuEventRecord(EvQueued, 0));
      UR_CHECK_ERROR(cuEventRecord(EvStart, Stream));
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }

  IsStarted = true;
  return Result;
}

bool ur_event_handle_t_::isCompleted() const noexcept {
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
}

uint64_t ur_event_handle_t_::getQueuedTime() const {
  assert(isStarted());
  return Queue->get_device()->getElapsedTime(EvQueued);
}

uint64_t ur_event_handle_t_::getStartTime() const {
  assert(isStarted());
  return Queue->get_device()->getElapsedTime(EvStart);
}

uint64_t ur_event_handle_t_::getEndTime() const {
  assert(isStarted() && isRecorded());
  return Queue->get_device()->getElapsedTime(EvEnd);
}

ur_result_t ur_event_handle_t_::record() {

  if (isRecorded() || !isStarted()) {
    return UR_RESULT_ERROR_INVALID_EVENT;
  }

  ur_result_t Result = UR_RESULT_SUCCESS;

  UR_ASSERT(Queue, UR_RESULT_ERROR_INVALID_QUEUE);

  try {
    EventID = Queue->getNextEventID();
    if (EventID == 0) {
      detail::ur::die(
          "Unrecoverable program state reached in event identifier overflow");
    }
    UR_CHECK_ERROR(cuEventRecord(EvEnd, Stream));
  } catch (ur_result_t error) {
    Result = error;
  }

  if (Result == UR_RESULT_SUCCESS) {
    IsRecorded = true;
  }

  return Result;
}

ur_result_t ur_event_handle_t_::wait() {
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    UR_CHECK_ERROR(cuEventSynchronize(EvEnd));
    HasBeenWaitedOn = true;
  } catch (ur_result_t error) {
    Result = error;
  }

  return Result;
}

ur_result_t ur_event_handle_t_::release() {
  if (!backendHasOwnership())
    return UR_RESULT_SUCCESS;

  assert(Queue != nullptr);

  UR_CHECK_ERROR(cuEventDestroy(EvEnd));

  if (Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE) {
    UR_CHECK_ERROR(cuEventDestroy(EvQueued));
    UR_CHECK_ERROR(cuEventDestroy(EvStart));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropValueSizeRet) {
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  switch (propName) {
  case UR_EVENT_INFO_COMMAND_QUEUE:
    return ReturnValue(hEvent->getQueue());
  case UR_EVENT_INFO_COMMAND_TYPE:
    return ReturnValue(hEvent->getCommandType());
  case UR_EVENT_INFO_REFERENCE_COUNT:
    return ReturnValue(hEvent->getReferenceCount());
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS:
    return ReturnValue(hEvent->getExecutionStatus());
  case UR_EVENT_INFO_CONTEXT:
    return ReturnValue(hEvent->getContext());
  default:
    detail::ur::die("Event info request not implemented");
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
  if (Queue == nullptr || !(Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE)) {
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
  default:
    break;
  }
  detail::ur::die("Event Profiling info request not implemented");
  return {};
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
    auto Context = phEventWaitList[0]->getContext();
    ScopedContext Active(Context);

    auto WaitFunc = [Context](ur_event_handle_t Event) -> ur_result_t {
      UR_ASSERT(Event, UR_RESULT_ERROR_INVALID_EVENT);
      UR_ASSERT(Event->getContext() == Context,
                UR_RESULT_ERROR_INVALID_CONTEXT);

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

  detail::ur::assertion(RefCount != 0,
                        "Reference count overflow detected in urEventRetain.");

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(ur_event_handle_t hEvent) {
  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  detail::ur::assertion(hEvent->getReferenceCount() != 0,
                        "Reference count overflow detected in urEventRelease.");

  // decrement ref count. If it is 0, delete the event.
  if (hEvent->decrementReferenceCount() == 0) {
    std::unique_ptr<ur_event_handle_t_> event_ptr{hEvent};
    ur_result_t Result = UR_RESULT_ERROR_INVALID_EVENT;
    try {
      ScopedContext Active(hEvent->getContext());
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
    const ur_event_native_properties_t *pProperties,
    ur_event_handle_t *phEvent) {
  std::ignore = pProperties;

  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  *phEvent = ur_event_handle_t_::makeWithNative(
      hContext, reinterpret_cast<CUevent>(hNativeEvent));

  return UR_RESULT_SUCCESS;
}
