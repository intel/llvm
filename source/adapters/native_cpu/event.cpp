//===--------- event.cpp - NATIVE CPU Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

#include "common.hpp"
#include "event.hpp"
#include "queue.hpp"
#include <cstdint>
#include <mutex>

UR_APIEXPORT ur_result_t UR_APICALL urEventGetInfo(ur_event_handle_t hEvent,
                                                   ur_event_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
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

UR_APIEXPORT ur_result_t UR_APICALL urEventGetProfilingInfo(
    ur_event_handle_t hEvent, ur_profiling_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_PROFILING_INFO_COMMAND_START:
    return ReturnValue(hEvent->get_start_timestamp());
  case UR_PROFILING_INFO_COMMAND_END:
    return ReturnValue(hEvent->get_end_timestamp());
  case UR_PROFILING_INFO_COMMAND_QUEUED:
  case UR_PROFILING_INFO_COMMAND_SUBMIT:
  case UR_PROFILING_INFO_COMMAND_COMPLETE:
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventWait(uint32_t numEvents, const ur_event_handle_t *phEventWaitList) {
  for (uint32_t i = 0; i < numEvents; i++) {
    phEventWaitList[i]->wait();
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRetain(ur_event_handle_t hEvent) {
  hEvent->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventRelease(ur_event_handle_t hEvent) {
  decrementOrDelete(hEvent);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventGetNativeHandle(
    ur_event_handle_t hEvent, ur_native_handle_t *phNativeEvent) {
  std::ignore = hEvent;
  std::ignore = phNativeEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEventCreateWithNativeHandle(
    ur_native_handle_t hNativeEvent, ur_context_handle_t hContext,
    const ur_event_native_properties_t *pProperties,
    ur_event_handle_t *phEvent) {
  std::ignore = hNativeEvent;
  std::ignore = hContext;
  std::ignore = pProperties;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEventSetCallback(ur_event_handle_t hEvent, ur_execution_info_t execStatus,
                   ur_event_callback_t pfnNotify, void *pUserData) {
  std::ignore = hEvent;
  std::ignore = execStatus;
  std::ignore = pfnNotify;
  std::ignore = pUserData;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueTimestampRecordingExp(
    ur_queue_handle_t hQueue, bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = blocking;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  DIE_NO_IMPLEMENTATION;
}

ur_event_handle_t_::ur_event_handle_t_(ur_queue_handle_t queue,
                                       ur_command_t command_type)
    : queue(queue), context(queue->getContext()), command_type(command_type),
      done(false) {
  this->queue->addEvent(this);
}

ur_event_handle_t_::~ur_event_handle_t_() {
  if (!done) {
    wait();
  }
}

void ur_event_handle_t_::wait() {
  std::unique_lock<std::mutex> lock(mutex);
  if (done) {
    return;
  }
  for (auto &f : futures) {
    f.wait();
  }
  queue->removeEvent(this);
  done = true;
  // The callback may need to acquire the lock, so we unlock it here
  lock.unlock();

  if (callback)
    callback();
}

void ur_event_handle_t_::tick_start() {
  if (!queue->isProfiling())
    return;
  std::lock_guard<std::mutex> lock(mutex);
  timestamp_start = get_timestamp();
}

void ur_event_handle_t_::tick_end() {
  if (!queue->isProfiling())
    return;
  std::lock_guard<std::mutex> lock(mutex);
  timestamp_end = get_timestamp();
}
