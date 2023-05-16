//===--------- event.cpp - HIP Adapter ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "event.hpp"
#include "common.hpp"
#include "context.hpp"
#include "platform.hpp"

ur_event_handle_t_::ur_event_handle_t_(ur_command_t type,
                                       ur_context_handle_t context,
                                       ur_queue_handle_t queue,
                                       hipStream_t stream,
                                       uint32_t stream_token)
    : commandType_{type}, refCount_{1}, hasBeenWaitedOn_{false},
      isRecorded_{false}, isStarted_{false},
      streamToken_{stream_token}, evEnd_{nullptr}, evStart_{nullptr},
      evQueued_{nullptr}, queue_{queue}, stream_{stream}, context_{context} {

  bool profilingEnabled = queue_->ur_flags_ & UR_QUEUE_FLAG_PROFILING_ENABLE;

  UR_CHECK_ERROR(hipEventCreateWithFlags(
      &evEnd_, profilingEnabled ? hipEventDefault : hipEventDisableTiming));

  if (profilingEnabled) {
    UR_CHECK_ERROR(hipEventCreateWithFlags(&evQueued_, hipEventDefault));
    UR_CHECK_ERROR(hipEventCreateWithFlags(&evStart_, hipEventDefault));
  }

  if (queue_ != nullptr) {
    urQueueRetain(queue_);
  }
  urContextRetain(context_);
}

ur_event_handle_t_::~ur_event_handle_t_() {
  if (queue_ != nullptr) {
    urQueueRelease(queue_);
  }
  urContextRelease(context_);
}

ur_result_t ur_event_handle_t_::start() {
  assert(!is_started());
  ur_result_t result = UR_RESULT_SUCCESS;

  try {
    if (queue_->ur_flags_ & UR_QUEUE_FLAG_PROFILING_ENABLE) {
      // NOTE: This relies on the default stream to be unused.
      UR_CHECK_ERROR(hipEventRecord(evQueued_, 0));
      UR_CHECK_ERROR(hipEventRecord(evStart_, queue_->get()));
    }
  } catch (ur_result_t error) {
    result = error;
  }

  isStarted_ = true;
  return result;
}

bool ur_event_handle_t_::is_completed() const noexcept {
  if (!isRecorded_) {
    return false;
  }
  if (!hasBeenWaitedOn_) {
    const hipError_t ret = hipEventQuery(evEnd_);
    if (ret != hipSuccess && ret != hipErrorNotReady) {
      UR_CHECK_ERROR(ret);
      return false;
    }
    if (ret == hipErrorNotReady) {
      return false;
    }
  }
  return true;
}

uint64_t ur_event_handle_t_::get_queued_time() const {
  float miliSeconds = 0.0f;
  assert(is_started());

  UR_CHECK_ERROR(hipEventElapsedTime(&miliSeconds, evStart_, evEnd_));
  return static_cast<uint64_t>(miliSeconds * 1.0e6);
}

uint64_t ur_event_handle_t_::get_start_time() const {
  float miliSeconds = 0.0f;
  assert(is_started());

  UR_CHECK_ERROR(hipEventElapsedTime(&miliSeconds,
                                     ur_platform_handle_t_::evBase_, evStart_));
  return static_cast<uint64_t>(miliSeconds * 1.0e6);
}

uint64_t ur_event_handle_t_::get_end_time() const {
  float miliSeconds = 0.0f;
  assert(is_started() && is_recorded());

  UR_CHECK_ERROR(hipEventElapsedTime(&miliSeconds,
                                     ur_platform_handle_t_::evBase_, evEnd_));
  return static_cast<uint64_t>(miliSeconds * 1.0e6);
}

ur_result_t ur_event_handle_t_::record() {

  if (is_recorded() || !is_started()) {
    return UR_RESULT_ERROR_INVALID_EVENT;
  }

  ur_result_t result = UR_RESULT_ERROR_INVALID_OPERATION;

  UR_ASSERT(queue_, UR_RESULT_ERROR_INVALID_QUEUE);

  try {
    eventId_ = queue_->get_next_event_id();
    if (eventId_ == 0) {
      sycl::detail::ur::die(
          "Unrecoverable program state reached in event identifier overflow");
    }
    result = UR_CHECK_ERROR(hipEventRecord(evEnd_, stream_));
  } catch (ur_result_t error) {
    result = error;
  }

  if (result == UR_RESULT_SUCCESS) {
    isRecorded_ = true;
  }

  return result;
}

ur_result_t ur_event_handle_t_::wait() {
  ur_result_t retErr;
  try {
    retErr = UR_CHECK_ERROR(hipEventSynchronize(evEnd_));
    hasBeenWaitedOn_ = true;
  } catch (ur_result_t error) {
    retErr = error;
  }

  return retErr;
}

ur_result_t ur_event_handle_t_::release() {
  assert(queue_ != nullptr);
  UR_CHECK_ERROR(hipEventDestroy(evEnd_));

  if (queue_->ur_flags_ & UR_QUEUE_FLAG_PROFILING_ENABLE) {
    UR_CHECK_ERROR(hipEventDestroy(evQueued_));
    UR_CHECK_ERROR(hipEventDestroy(evStart_));
  }

  return UR_RESULT_SUCCESS;
}

////////////////////

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  UR_ASSERT(numEvents > 0, UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(phEventWaitList, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  try {

    auto context = phEventWaitList[0]->get_context();
    ScopedContext active(context);

    auto waitFunc = [context](ur_event_handle_t event) -> ur_result_t {
      UR_ASSERT(event, UR_RESULT_ERROR_INVALID_EVENT);
      UR_ASSERT(event->get_context() == context,
                UR_RESULT_ERROR_INVALID_CONTEXT);

      return event->wait();
    };
    return forLatestEvents(phEventWaitList, numEvents, waitFunc);
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

//
// Events
//

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropValueSizeRet) {
  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(!(pPropValue && propValueSize == 0), UR_RESULT_ERROR_INVALID_SIZE);

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);
  switch (propName) {
  case UR_EVENT_INFO_COMMAND_QUEUE:
    return ReturnValue(hEvent->get_queue());
  case UR_EVENT_INFO_COMMAND_TYPE:
    return ReturnValue(hEvent->get_command_type());
  case UR_EVENT_INFO_REFERENCE_COUNT:
    return ReturnValue(hEvent->get_reference_count());
  case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS:
    return ReturnValue(hEvent->get_execution_status());
  case UR_EVENT_INFO_CONTEXT:
    return ReturnValue(hEvent->get_context());
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

  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(!(pPropValue && propValueSize == 0), UR_RESULT_ERROR_INVALID_VALUE);

  ur_queue_handle_t queue = hEvent->get_queue();
  if (queue == nullptr ||
      !(queue->ur_flags_ & UR_QUEUE_FLAG_PROFILING_ENABLE)) {
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);
  switch (propName) {
  case UR_PROFILING_INFO_COMMAND_QUEUED:
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
    // Note: No user for this case
    return ReturnValue(static_cast<uint64_t>(hEvent->get_queued_time()));
  case UR_PROFILING_INFO_COMMAND_START:
    return ReturnValue(static_cast<uint64_t>(hEvent->get_start_time()));
  case UR_PROFILING_INFO_COMMAND_END:
    return ReturnValue(static_cast<uint64_t>(hEvent->get_end_time()));
  default:
    break;
  }
  return {};
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventSetCallback(ur_event_handle_t hEvent, ur_execution_info_t execStatus,
                   ur_event_callback_t pfnNotify, void *pUserData) {
  std::ignore = hEvent;
  std::ignore = execStatus;
  std::ignore = pfnNotify;
  std::ignore = pUserData;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(ur_event_handle_t hEvent) {
  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  const auto refCount = hEvent->increment_reference_count();

  sycl::detail::ur::assertion(
      refCount != 0, "Reference count overflow detected in urEventRetain.");

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(ur_event_handle_t hEvent) {
  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  sycl::detail::ur::assertion(
      hEvent->get_reference_count() != 0,
      "Reference count overflow detected in urEventRelease.");

  // decrement ref count. If it is 0, delete the event.
  if (hEvent->decrement_reference_count() == 0) {
    std::unique_ptr<ur_event_handle_t_> event_ptr{hEvent};
    ur_result_t result = UR_RESULT_ERROR_INVALID_EVENT;
    try {
      ScopedContext active(hEvent->get_context());
      result = hEvent->release();
    } catch (...) {
      result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }
    return result;
  }

  return UR_RESULT_SUCCESS;
}

/// Gets the native HIP handle of a UR event object
///
/// \param[in] event The UR event to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the UR event object.
///
/// \return UR_RESULT_SUCCESS on success. UR_RESULT_ERROR_INVALID_EVENT if given
/// a user event.
UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ur_native_handle_t *phNativeEvent) {
  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeEvent, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeEvent = reinterpret_cast<ur_native_handle_t>(hEvent->get());
  return UR_RESULT_SUCCESS;
}

/// Created a UR event object from a HIP event handle.
/// TODO: Implement this.
/// NOTE: The created UR object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create UR event object from.
/// \param[out] event Set to the UR event object created from native handle.
///
/// \return TBD
UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ur_context_handle_t hContext,
    const ur_event_native_properties_t *pProperties,
    ur_event_handle_t *phEvent) {

  std::ignore = hNativeEvent;
  std::ignore = hContext;
  std::ignore = pProperties;
  std::ignore = phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
