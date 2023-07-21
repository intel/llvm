//===----------- queue.cpp - Native CPU Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
  std::ignore = hQueue;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *pProperties, ur_queue_handle_t *phQueue) {
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pProperties;

  auto Queue = new ur_queue_handle_t_();
  *phQueue = Queue;

  CONTINUE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  std::ignore = hQueue;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  delete hQueue;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetNativeHandle(ur_queue_handle_t hQueue, ur_queue_native_desc_t *pDesc,
                       ur_native_handle_t *phNativeQueue) {
  std::ignore = hQueue;
  std::ignore = pDesc;
  std::ignore = phNativeQueue;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {
  std::ignore = hNativeQueue;
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pProperties;
  std::ignore = phQueue;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  std::ignore = hQueue;
  // TODO: is this fine as no-op?
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t hQueue) {
  std::ignore = hQueue;

  DIE_NO_IMPLEMENTATION;
}
