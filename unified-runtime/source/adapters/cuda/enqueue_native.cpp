//===--------- enqueue_native.cpp - CUDA Adapter --------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur_api.h>

#include "context.hpp"
#include "event.hpp"
#include "memory.hpp"
#include "queue.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueNativeCommandExp(
    ur_queue_handle_t hQueue,
    ur_exp_enqueue_native_command_function_t pfnNativeEnqueue, void *data,
    uint32_t NumMemsInMemList, const ur_mem_handle_t *phMemList,
    const ur_exp_enqueue_native_command_properties_t *,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  try {
    ScopedContext ActiveContext(hQueue->getDevice());
    InteropGuard ActiveStream(hQueue, NumEventsInWaitList, phEventWaitList);
    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

    if (hQueue->getContext()->getDevices().size() > 1) {
      for (auto i = 0u; i < NumMemsInMemList; ++i) {
        enqueueMigrateMemoryToDeviceIfNeeded(phMemList[i], hQueue->getDevice(),
                                             ActiveStream.getStream());
        phMemList[i]->setLastQueueWritingToMemObj(hQueue);
      }
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_ENQUEUE_NATIVE_EXP, hQueue, ActiveStream.getStream()));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    pfnNativeEnqueue(hQueue, data); // This is using urQueueGetNativeHandle to
                                    // get the CUDA stream. It must be the
                                    // same stream as is used before and after

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (CUresult CuErr) {
    return mapErrorUR(CuErr);
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}
