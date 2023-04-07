//===--------- enqueue.cpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "queue.hpp"

#include <cuda.h>

/// Enqueues a wait on the given CUstream for all specified events (See
/// \ref enqueueEventWaitWithBarrier.) If the events list is empty, the enqueued
/// wait will wait on all previous events in the queue.
///
ur_result_t urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // This function makes one stream work on the previous work (or work
  // represented by input events) and then all future work waits on that stream.
  if (!hQueue) {
    return UR_RESULT_ERROR_INVALID_QUEUE;
  }

  ur_result_t result;

  try {
    ScopedContext active(hQueue->get_context());
    uint32_t stream_token;
    ur_stream_guard_ guard;
    CUstream cuStream = hQueue->get_next_compute_stream(
        numEventsInWaitList, phEventWaitList, guard, &stream_token);
    {
      std::lock_guard<std::mutex> guard(hQueue->barrier_mutex_);
      if (hQueue->barrier_event_ == nullptr) {
        UR_CHECK_ERROR(
            cuEventCreate(&hQueue->barrier_event_, CU_EVENT_DISABLE_TIMING));
      }
      if (numEventsInWaitList == 0) { //  wait on all work
        if (hQueue->barrier_tmp_event_ == nullptr) {
          UR_CHECK_ERROR(cuEventCreate(&hQueue->barrier_tmp_event_,
                                       CU_EVENT_DISABLE_TIMING));
        }
        hQueue->sync_streams(
            [cuStream, tmp_event = hQueue->barrier_tmp_event_](CUstream s) {
              if (cuStream != s) {
                // record a new CUDA event on every stream and make one stream
                // wait for these events
                UR_CHECK_ERROR(cuEventRecord(tmp_event, s));
                UR_CHECK_ERROR(cuStreamWaitEvent(cuStream, tmp_event, 0));
              }
            });
      } else { // wait just on given events
        forLatestEvents(phEventWaitList, numEventsInWaitList,
                        [cuStream](ur_event_handle_t event) -> ur_result_t {
                          if (event->get_queue()->has_been_synchronized(
                                  event->get_compute_stream_token())) {
                            return UR_RESULT_SUCCESS;
                          } else {
                            return UR_CHECK_ERROR(
                                cuStreamWaitEvent(cuStream, event->get(), 0));
                          }
                        });
      }

      result = UR_CHECK_ERROR(cuEventRecord(hQueue->barrier_event_, cuStream));
      for (unsigned int i = 0; i < hQueue->compute_applied_barrier_.size();
           i++) {
        hQueue->compute_applied_barrier_[i] = false;
      }
      for (unsigned int i = 0; i < hQueue->transfer_applied_barrier_.size();
           i++) {
        hQueue->transfer_applied_barrier_[i] = false;
      }
    }
    if (result != UR_RESULT_SUCCESS) {
      return result;
    }

    if (phEvent) {
      *phEvent = ur_event_handle_t_::make_native(
          UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, hQueue, cuStream, stream_token);
      (*phEvent)->start();
      (*phEvent)->record();
    }

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

/// Enqueues a wait on the given CUstream for all events.
/// See \ref enqueueEventWait
/// TODO: Add support for multiple streams once the Event class is properly
/// refactored.
///
ur_result_t urEnqueueEventsWait(ur_queue_handle_t hQueue,
                                uint32_t numEventsInWaitList,
                                const ur_event_handle_t *phEventWaitList,
                                ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}
