//===--------- async_alloc.cpp - CUDA Adapter -----------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "context.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "queue.hpp"
#include "usm.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMDeviceAllocExp(
    ur_queue_handle_t hQueue, ur_usm_pool_handle_t hPool, const size_t size,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, void **ppMem,
    ur_event_handle_t *phEvent) try {
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  ScopedContext Active(hQueue->getDevice());
  uint32_t StreamToken;
  ur_stream_guard_ Guard;
  CUstream CuStream = hQueue->getNextComputeStream(
      numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

  UR_CHECK_ERROR(enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                                   phEventWaitList));

  if (phEvent) {
    RetImplEvent =
        std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
            UR_COMMAND_KERNEL_LAUNCH, hQueue, CuStream, StreamToken));
    UR_CHECK_ERROR(RetImplEvent->start());
  }

  // Allocate to the device pool if specified. Otherwise, allocate to the
  // current device pool.
  if (hPool) {
    assert(hPool->usesCudaPool());
    UR_CHECK_ERROR(
        cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr *>(ppMem), size,
                                hPool->getCudaPool(), CuStream));
  } else {
    UR_CHECK_ERROR(cuMemAllocAsync(reinterpret_cast<CUdeviceptr *>(ppMem), size,
                                   CuStream));
  }

  if (phEvent) {
    UR_CHECK_ERROR(RetImplEvent->record());
    *phEvent = RetImplEvent.release();
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMHostAllocExp(
    ur_queue_handle_t, ur_usm_pool_handle_t, const size_t,
    const ur_exp_async_usm_alloc_properties_t *, uint32_t,
    const ur_event_handle_t *, void **, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMSharedAllocExp(
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

  ScopedContext Active(hQueue->getDevice());
  uint32_t StreamToken;
  ur_stream_guard_ Guard;
  CUstream CuStream = hQueue->getNextComputeStream(
      numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

  UR_CHECK_ERROR(enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                                   phEventWaitList));

  if (phEvent) {
    RetImplEvent =
        std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
            UR_COMMAND_KERNEL_LAUNCH, hQueue, CuStream, StreamToken));
    UR_CHECK_ERROR(RetImplEvent->start());
  }

  UR_CHECK_ERROR(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(pMem), CuStream));

  if (phEvent) {
    UR_CHECK_ERROR(RetImplEvent->record());
    *phEvent = RetImplEvent.release();
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}
