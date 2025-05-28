//===----------- queue.cpp - Native CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue.hpp"
#include "common.hpp"

#include "ur/ur.hpp"
#include "ur_api.h"

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_QUEUE_INFO_CONTEXT:
    return ReturnValue(hQueue->getContext());
  case UR_QUEUE_INFO_DEVICE:
    return ReturnValue(hQueue->getDevice());
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(hQueue->getReferenceCount());
  case UR_QUEUE_INFO_EMPTY:
    return ReturnValue(hQueue->isEmpty());
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProperties, ur_queue_handle_t *phQueue) {
  // TODO: UR_QUEUE_FLAG_PROFILING_ENABLE and other props

  auto Queue = new ur_queue_handle_t_(hDevice, hContext, pProperties);
  *phQueue = Queue;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  hQueue->incrementReferenceCount();

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  decrementOrDelete(hQueue);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t /*hQueue*/, ur_queue_native_desc_t * /*pDesc*/,
    ur_native_handle_t * /*phNativeQueue*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t /*hNativeQueue*/, ur_context_handle_t /*hContext*/,
    ur_device_handle_t /*hDevice*/,
    const ur_queue_native_properties_t * /*pProperties*/,
    ur_queue_handle_t * /*phQueue*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  hQueue->finish();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t /*hQueue*/) {

  DIE_NO_IMPLEMENTATION;
}
