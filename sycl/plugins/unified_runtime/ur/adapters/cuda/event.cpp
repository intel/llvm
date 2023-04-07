//===--------- event.cpp - CUDA Adapter ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "event.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "queue.hpp"

#include <cassert>
#include <cuda.h>

ur_event_handle_t_::ur_event_handle_t_(ur_command_t type,
                                       ur_context_handle_t context,
                                       ur_queue_handle_t queue, CUstream stream,
                                       uint32_t stream_token)
    : commandType_{type}, refCount_{1}, has_ownership_{true},
      hasBeenWaitedOn_{false}, isRecorded_{false}, isStarted_{false},
      streamToken_{stream_token}, evEnd_{nullptr}, evStart_{nullptr},
      evQueued_{nullptr}, queue_{queue}, stream_{stream}, context_{context} {

  bool profilingEnabled = queue_->ur_flags_ & UR_QUEUE_FLAG_PROFILING_ENABLE;

  UR_CHECK_ERROR(cuEventCreate(
      &evEnd_, profilingEnabled ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING));

  if (profilingEnabled) {
    UR_CHECK_ERROR(cuEventCreate(&evQueued_, CU_EVENT_DEFAULT));
    UR_CHECK_ERROR(cuEventCreate(&evStart_, CU_EVENT_DEFAULT));
  }

  if (queue_ != nullptr) {
    urQueueRetain(queue_);
  }
  urContextRetain(context_);
}

ur_event_handle_t_::ur_event_handle_t_(ur_context_handle_t context,
                                       CUevent eventNative)
    // TODO(ur): Missing user command type
    : commandType_{UR_COMMAND_EVENTS_WAIT}, refCount_{1}, has_ownership_{false},
      hasBeenWaitedOn_{false}, isRecorded_{false}, isStarted_{false},
      streamToken_{std::numeric_limits<uint32_t>::max()}, evEnd_{eventNative},
      evStart_{nullptr}, evQueued_{nullptr}, queue_{nullptr}, context_{
                                                                  context} {
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
      result = UR_CHECK_ERROR(cuEventRecord(evQueued_, 0));
      result = UR_CHECK_ERROR(cuEventRecord(evStart_, stream_));
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
    const CUresult ret = cuEventQuery(evEnd_);
    if (ret != CUDA_SUCCESS && ret != CUDA_ERROR_NOT_READY) {
      UR_CHECK_ERROR(ret);
      return false;
    }
    if (ret == CUDA_ERROR_NOT_READY) {
      return false;
    }
  }
  return true;
}

uint64_t ur_event_handle_t_::get_queued_time() const {
  assert(is_started());
  return queue_->get_device()->get_elapsed_time(evQueued_);
}

uint64_t ur_event_handle_t_::get_start_time() const {
  assert(is_started());
  return queue_->get_device()->get_elapsed_time(evStart_);
}

uint64_t ur_event_handle_t_::get_end_time() const {
  assert(is_started() && is_recorded());
  return queue_->get_device()->get_elapsed_time(evEnd_);
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
    result = UR_CHECK_ERROR(cuEventRecord(evEnd_, stream_));
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
    retErr = UR_CHECK_ERROR(cuEventSynchronize(evEnd_));
    hasBeenWaitedOn_ = true;
  } catch (ur_result_t error) {
    retErr = error;
  }

  return retErr;
}

ur_result_t ur_event_handle_t_::release() {
  if (!backend_has_ownership())
    return UR_RESULT_SUCCESS;

  assert(queue_ != nullptr);

  UR_CHECK_ERROR(cuEventDestroy(evEnd_));

  if (queue_->ur_flags_ & UR_QUEUE_FLAG_PROFILING_ENABLE) {
    UR_CHECK_ERROR(cuEventDestroy(evQueued_));
    UR_CHECK_ERROR(cuEventDestroy(evStart_));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropValueSizeRet) {
  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
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
    sycl::detail::ur::die("Event info request not implemented");
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

/// Obtain profiling information from PI CUDA events
/// \TODO Timings from CUDA are only elapsed time.
UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ur_profiling_info_t propName,
    size_t propValueSize, void *pPropValue, size_t *pPropValueSizeRet) {
  UR_ASSERT(hEvent, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  ur_queue_handle_t queue = hEvent->get_queue();
  if (queue == nullptr ||
      !(queue->ur_flags_ & UR_QUEUE_FLAG_PROFILING_ENABLE)) {
    return UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE;
  }

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
  sycl::detail::ur::die("Event Profiling info request not implemented");
  return {};
}

UR_APIEXPORT ur_result_t UR_APICALL urEventSetCallback(ur_event_handle_t,
                                                       ur_execution_info_t,
                                                       ur_event_callback_t,
                                                       void *) {
  sycl::detail::ur::die("Event Callback not implemented in CUDA adapter");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  try {
    UR_ASSERT(phEventWaitList, UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
    UR_ASSERT(numEvents > 0, UR_RESULT_ERROR_INVALID_VALUE);

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

UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ur_native_handle_t *phNativeEvent) {
  *phNativeEvent = reinterpret_cast<ur_native_handle_t>(hEvent->get());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ur_context_handle_t hContext,
    const ur_event_native_properties_t *pProperties,
    ur_event_handle_t *phEvent) {
  (void)pProperties;

  std::unique_ptr<ur_event_handle_t_> event_ptr{nullptr};

  *phEvent = ur_event_handle_t_::make_with_native(
      hContext, reinterpret_cast<CUevent>(hNativeEvent));

  return UR_RESULT_SUCCESS;
}
