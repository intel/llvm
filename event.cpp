//===--------- event.cpp - HIP Adapter ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "event.hpp"
#include "common.hpp"
#include "context.hpp"
#include "platform.hpp"

ur_event_handle_t_::ur_event_handle_t_(ur_command_t Type,
                                       ur_context_handle_t Context,
                                       ur_queue_handle_t Queue,
                                       hipStream_t Stream, uint32_t StreamToken)
    : CommandType{Type}, RefCount{1}, HasOwnership{true},
      HasBeenWaitedOn{false}, IsRecorded{false}, IsStarted{false},
      StreamToken{StreamToken}, EvEnd{nullptr}, EvStart{nullptr},
      EvQueued{nullptr}, Queue{Queue}, Stream{Stream}, Context{Context} {

  bool ProfilingEnabled = Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE;

  UR_CHECK_ERROR(hipEventCreateWithFlags(
      &EvEnd, ProfilingEnabled ? hipEventDefault : hipEventDisableTiming));

  if (ProfilingEnabled) {
    UR_CHECK_ERROR(hipEventCreateWithFlags(&EvQueued, hipEventDefault));
    UR_CHECK_ERROR(hipEventCreateWithFlags(&EvStart, hipEventDefault));
  }

  if (Queue != nullptr) {
    urQueueRetain(Queue);
  }
  urContextRetain(Context);
}

ur_event_handle_t_::ur_event_handle_t_(ur_context_handle_t Context,
                                       hipEvent_t EventNative)
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
      UR_CHECK_ERROR(hipEventRecord(EvQueued, 0));
      UR_CHECK_ERROR(hipEventRecord(EvStart, Queue->get()));
    }
  } catch (ur_result_t Error) {
    Result = Error;
  }

  IsStarted = true;
  return Result;
}

bool ur_event_handle_t_::isCompleted() const noexcept {
  if (!IsRecorded) {
    return false;
  }
  if (!HasBeenWaitedOn) {
    const hipError_t Result = hipEventQuery(EvEnd);
    if (Result != hipSuccess && Result != hipErrorNotReady) {
      UR_CHECK_ERROR(Result);
      return false;
    }
    if (Result == hipErrorNotReady) {
      return false;
    }
  }
  return true;
}

uint64_t ur_event_handle_t_::getQueuedTime() const {
  float MilliSeconds = 0.0f;
  assert(isStarted());

  // hipEventSynchronize waits till the event is ready for call to
  // hipEventElapsedTime.
  UR_CHECK_ERROR(hipEventSynchronize(EvStart));
  UR_CHECK_ERROR(hipEventSynchronize(EvEnd));

  UR_CHECK_ERROR(hipEventElapsedTime(&MilliSeconds, EvStart, EvEnd));
  return static_cast<uint64_t>(MilliSeconds * 1.0e6);
}

uint64_t ur_event_handle_t_::getStartTime() const {
  float MiliSeconds = 0.0f;
  assert(isStarted());

  // hipEventSynchronize waits till the event is ready for call to
  // hipEventElapsedTime.
  UR_CHECK_ERROR(hipEventSynchronize(ur_platform_handle_t_::EvBase));
  UR_CHECK_ERROR(hipEventSynchronize(EvStart));

  UR_CHECK_ERROR(hipEventElapsedTime(&MiliSeconds,
                                     ur_platform_handle_t_::EvBase, EvStart));
  return static_cast<uint64_t>(MiliSeconds * 1.0e6);
}

uint64_t ur_event_handle_t_::getEndTime() const {
  float MiliSeconds = 0.0f;
  assert(isStarted() && isRecorded());

  // hipEventSynchronize waits till the event is ready for call to
  // hipEventElapsedTime.
  UR_CHECK_ERROR(hipEventSynchronize(ur_platform_handle_t_::EvBase));
  UR_CHECK_ERROR(hipEventSynchronize(EvEnd));

  UR_CHECK_ERROR(
      hipEventElapsedTime(&MiliSeconds, ur_platform_handle_t_::EvBase, EvEnd));
  return static_cast<uint64_t>(MiliSeconds * 1.0e6);
}

ur_result_t ur_event_handle_t_::record() {

  if (isRecorded() || !isStarted()) {
    return UR_RESULT_ERROR_INVALID_EVENT;
  }

  ur_result_t Result = UR_RESULT_ERROR_INVALID_OPERATION;

  UR_ASSERT(Queue, UR_RESULT_ERROR_INVALID_QUEUE);

  try {
    EventId = Queue->getNextEventId();
    if (EventId == 0) {
      detail::ur::die(
          "Unrecoverable program state reached in event identifier overflow");
    }
    UR_CHECK_ERROR(hipEventRecord(EvEnd, Stream));
    Result = UR_RESULT_SUCCESS;
  } catch (ur_result_t Error) {
    Result = Error;
  }

  if (Result == UR_RESULT_SUCCESS) {
    IsRecorded = true;
  }

  return Result;
}

ur_result_t ur_event_handle_t_::wait() {
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    UR_CHECK_ERROR(hipEventSynchronize(EvEnd));
    HasBeenWaitedOn = true;
  } catch (ur_result_t Error) {
    Result = Error;
  }

  return Result;
}

ur_result_t ur_event_handle_t_::release() {
  if (!backendHasOwnership())
    return UR_RESULT_SUCCESS;

  assert(Queue != nullptr);
  UR_CHECK_ERROR(hipEventDestroy(EvEnd));

  if (Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE) {
    UR_CHECK_ERROR(hipEventDestroy(EvQueued));
    UR_CHECK_ERROR(hipEventDestroy(EvStart));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  UR_ASSERT(numEvents > 0, UR_RESULT_ERROR_INVALID_VALUE);

  try {

    auto Context = phEventWaitList[0]->getContext();
    ScopedContext Active(Context->getDevice());

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

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropValueSizeRet) {
  UR_ASSERT(!(pPropValue && propValueSize == 0), UR_RESULT_ERROR_INVALID_SIZE);

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
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

/// Obtain profiling information from UR HIP events
/// Timings from HIP are only elapsed time.
UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ur_profiling_info_t propName,
    size_t propValueSize, void *pPropValue, size_t *pPropValueSizeRet) {

  UR_ASSERT(!(pPropValue && propValueSize == 0), UR_RESULT_ERROR_INVALID_VALUE);

  ur_queue_handle_t Queue = hEvent->getQueue();
  if (Queue == nullptr || !(Queue->URFlags & UR_QUEUE_FLAG_PROFILING_ENABLE)) {
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);
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
  return {};
}

UR_APIEXPORT ur_result_t UR_APICALL urEventSetCallback(ur_event_handle_t,
                                                       ur_execution_info_t,
                                                       ur_event_callback_t,
                                                       void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
      ScopedContext Active(hEvent->getContext()->getDevice());
      Result = hEvent->release();
    } catch (...) {
      Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }
    return Result;
  }

  return UR_RESULT_SUCCESS;
}

/// Gets the native HIP handle of a UR event object
///
/// \param[in] hEvent The UR event to get the native HIP object of.
/// \param[out] phNativeEvent Set to the native handle of the UR event object.
///
/// \return UR_RESULT_SUCCESS on success. UR_RESULT_ERROR_INVALID_EVENT if given
/// a user event.
UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ur_native_handle_t *phNativeEvent) {
  *phNativeEvent = reinterpret_cast<ur_native_handle_t>(hEvent->get());
  return UR_RESULT_SUCCESS;
}

/// Created a UR event object from a HIP event handle.
/// NOTE: The created UR object doesn't take ownership of the native handle.
///
/// \param[in] hNativeEvent The native handle to create UR event object from.
/// \param[out] phEvent Set to the UR event object created from native handle.
UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ur_context_handle_t hContext,
    const ur_event_native_properties_t *pProperties,
    ur_event_handle_t *phEvent) {
  std::ignore = pProperties;

  *phEvent = ur_event_handle_t_::makeWithNative(
      hContext, reinterpret_cast<hipEvent_t>(hNativeEvent));

  return UR_RESULT_SUCCESS;
}
