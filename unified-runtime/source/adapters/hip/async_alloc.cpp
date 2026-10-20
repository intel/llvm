//===--------- async_alloc.cpp - HIP Adapter ------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unified-runtime/ur_api.h>

#include "common.hpp"
#include "context.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "queue.hpp"
#include "usm.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMDeviceAllocExp(
    ur_queue_handle_t hQueue, [[maybe_unused]] ur_usm_pool_handle_t hPool,
    const size_t size, const ur_exp_async_usm_alloc_properties_t *,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    void **ppMem, ur_event_handle_t *phEvent) try {
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  ScopedDevice Active(hQueue->getDevice());
  uint32_t StreamToken;
  ur_stream_guard Guard;
  hipStream_t HIPStream = hQueue->getNextComputeStream(
      numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

  UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                   phEventWaitList));

  if (phEvent) {
    RetImplEvent = std::make_unique<ur_event_handle_t_>(
        UR_COMMAND_ENQUEUE_USM_DEVICE_ALLOC_EXP, hQueue, HIPStream,
        StreamToken);
    UR_CHECK_ERROR(RetImplEvent->start());
  }

  // Allocate from the device's default stream-ordered memory pool. The HIP
  // adapter does not expose a native pool for ur_usm_pool_handle_t, so any
  // provided pool falls back to the default pool.
  UR_CHECK_ERROR(hipMallocAsync(ppMem, size, HIPStream));

  if (phEvent) {
    UR_CHECK_ERROR(RetImplEvent->record());
    *phEvent = RetImplEvent.release();
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMSharedAllocExp(
    ur_queue_handle_t, ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMHostAllocExp(
    ur_queue_handle_t, ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFreeExp(
    ur_queue_handle_t hQueue, [[maybe_unused]] ur_usm_pool_handle_t hPool,
    void *pMem, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) try {
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  ScopedDevice Active(hQueue->getDevice());
  uint32_t StreamToken;
  ur_stream_guard Guard;
  hipStream_t HIPStream = hQueue->getNextComputeStream(
      numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

  UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                   phEventWaitList));

  if (phEvent) {
    RetImplEvent = std::make_unique<ur_event_handle_t_>(
        UR_COMMAND_ENQUEUE_USM_FREE_EXP, hQueue, HIPStream, StreamToken);
    UR_CHECK_ERROR(RetImplEvent->start());
  }

  UR_CHECK_ERROR(hipFreeAsync(pMem, HIPStream));

  if (phEvent) {
    UR_CHECK_ERROR(RetImplEvent->record());
    *phEvent = RetImplEvent.release();
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}
