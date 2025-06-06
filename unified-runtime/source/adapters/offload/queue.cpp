//===----------- queue.cpp - LLVM Offload Adapter  ------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

#include "context.hpp"
#include "device.hpp"
#include "queue.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreate(
    [[maybe_unused]] ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_queue_properties_t *, ur_queue_handle_t *phQueue) {

  assert(hContext->Device == hDevice);

  ur_queue_handle_t Queue = new ur_queue_handle_t_();
  auto Res = olCreateQueue(hDevice->OffloadDevice, &Queue->OffloadQueue);
  if (Res != OL_SUCCESS) {
    delete Queue;
    return offloadResultToUR(Res);
  }

  Queue->OffloadDevice = hDevice->OffloadDevice;

  *phQueue = Queue;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(hQueue->RefCount.load());
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  hQueue->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  if (--hQueue->RefCount == 0) {
    auto Res = olDestroyQueue(hQueue->OffloadQueue);
    if (Res) {
      return offloadResultToUR(Res);
    }
    delete hQueue;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  return offloadResultToUR(olWaitQueue(hQueue->OffloadQueue));
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t, ur_queue_native_desc_t *, ur_native_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t, ur_device_handle_t,
    const ur_queue_native_properties_t *, ur_queue_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
