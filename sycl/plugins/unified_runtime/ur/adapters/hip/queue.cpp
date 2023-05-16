//===--------- queue.cpp - HIP Adapter ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "queue.hpp"
#include "context.hpp"
#include "event.hpp"

void ur_queue_handle_t_::compute_stream_wait_for_barrier_if_needed(
    hipStream_t stream, uint32_t stream_i) {
  if (barrier_event_ && !compute_applied_barrier_[stream_i]) {
    UR_CHECK_ERROR(hipStreamWaitEvent(stream, barrier_event_, 0));
    compute_applied_barrier_[stream_i] = true;
  }
}

void ur_queue_handle_t_::transfer_stream_wait_for_barrier_if_needed(
    hipStream_t stream, uint32_t stream_i) {
  if (barrier_event_ && !transfer_applied_barrier_[stream_i]) {
    UR_CHECK_ERROR(hipStreamWaitEvent(stream, barrier_event_, 0));
    transfer_applied_barrier_[stream_i] = true;
  }
}

hipStream_t
ur_queue_handle_t_::get_next_compute_stream(uint32_t *stream_token) {
  uint32_t stream_i;
  uint32_t token;
  while (true) {
    if (num_compute_streams_ < compute_streams_.size()) {
      // the check above is for performance - so as not to lock mutex every time
      std::lock_guard<std::mutex> guard(compute_stream_mutex_);
      // The second check is done after mutex is locked so other threads can not
      // change num_compute_streams_ after that
      if (num_compute_streams_ < compute_streams_.size()) {
        UR_CHECK_ERROR(hipStreamCreateWithFlags(
            &compute_streams_[num_compute_streams_++], flags_));
      }
    }
    token = compute_stream_idx_++;
    stream_i = token % compute_streams_.size();
    // if a stream has been reused before it was next selected round-robin
    // fashion, we want to delay its next use and instead select another one
    // that is more likely to have completed all the enqueued work.
    if (delay_compute_[stream_i]) {
      delay_compute_[stream_i] = false;
    } else {
      break;
    }
  }
  if (stream_token) {
    *stream_token = token;
  }
  hipStream_t res = compute_streams_[stream_i];
  compute_stream_wait_for_barrier_if_needed(res, stream_i);
  return res;
}

hipStream_t ur_queue_handle_t_::get_next_compute_stream(
    uint32_t num_events_in_wait_list, const ur_event_handle_t *event_wait_list,
    ur_stream_quard &guard, uint32_t *stream_token) {
  for (uint32_t i = 0; i < num_events_in_wait_list; i++) {
    uint32_t token = event_wait_list[i]->get_compute_stream_token();
    if (event_wait_list[i]->get_queue() == this && can_reuse_stream(token)) {
      std::unique_lock<std::mutex> compute_sync_guard(
          compute_stream_sync_mutex_);
      // redo the check after lock to avoid data races on
      // last_sync_compute_streams_
      if (can_reuse_stream(token)) {
        uint32_t stream_i = token % delay_compute_.size();
        delay_compute_[stream_i] = true;
        if (stream_token) {
          *stream_token = token;
        }
        guard = ur_stream_quard{std::move(compute_sync_guard)};
        hipStream_t res = event_wait_list[i]->get_stream();
        compute_stream_wait_for_barrier_if_needed(res, stream_i);
        return res;
      }
    }
  }
  guard = {};
  return get_next_compute_stream(stream_token);
}

hipStream_t ur_queue_handle_t_::get_next_transfer_stream() {
  if (transfer_streams_.empty()) { // for example in in-order queue
    return get_next_compute_stream();
  }
  if (num_transfer_streams_ < transfer_streams_.size()) {
    // the check above is for performance - so as not to lock mutex every time
    std::lock_guard<std::mutex> guard(transfer_stream_mutex_);
    // The second check is done after mutex is locked so other threads can not
    // change num_transfer_streams_ after that
    if (num_transfer_streams_ < transfer_streams_.size()) {
      UR_CHECK_ERROR(hipStreamCreateWithFlags(
          &transfer_streams_[num_transfer_streams_++], flags_));
    }
  }
  uint32_t stream_i = transfer_stream_idx_++ % transfer_streams_.size();
  hipStream_t res = transfer_streams_[stream_i];
  transfer_stream_wait_for_barrier_if_needed(res, stream_i);
  return res;
}

///////////////////////////////

UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
              const ur_queue_properties_t *pProps, ur_queue_handle_t *phQueue) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phQueue, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  try {
    std::unique_ptr<ur_queue_handle_t_> queueImpl{nullptr};

    if (hContext->get_device() != hDevice) {
      *phQueue = nullptr;
      return UR_RESULT_ERROR_INVALID_DEVICE;
    }

    unsigned int flags = 0;

    const bool is_out_of_order =
        pProps->flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    std::vector<hipStream_t> computeHipStreams(
        is_out_of_order ? ur_queue_handle_t_::default_num_compute_streams : 1);
    std::vector<hipStream_t> transferHipStreams(
        is_out_of_order ? ur_queue_handle_t_::default_num_transfer_streams : 0);

    queueImpl = std::unique_ptr<ur_queue_handle_t_>(new ur_queue_handle_t_{
        std::move(computeHipStreams), std::move(transferHipStreams), hContext,
        hDevice, flags, pProps->flags});

    *phQueue = queueImpl.release();

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t err) {

    return err;

  } catch (...) {

    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_QUEUE_INFO_CONTEXT:
    return ReturnValue(hQueue->context_);
  case UR_QUEUE_INFO_DEVICE:
    return ReturnValue(hQueue->device_);
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(hQueue->get_reference_count());
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(hQueue->ur_flags_);
  case UR_QUEUE_INFO_EMPTY: {
    bool IsReady = hQueue->all_of([](hipStream_t s) -> bool {
      const hipError_t ret = hipStreamQuery(s);
      if (ret == hipSuccess)
        return true;

      try {
        UR_CHECK_ERROR(ret);
      } catch (...) {
        return false;
      }

      return false;
    });
    return ReturnValue(IsReady);
  }
  default:
    break;
  }
  return {};
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hQueue->get_reference_count() > 0, UR_RESULT_ERROR_INVALID_QUEUE);

  hQueue->increment_reference_count();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  if (hQueue->decrement_reference_count() > 0) {
    return UR_RESULT_SUCCESS;
  }

  try {
    std::unique_ptr<ur_queue_handle_t_> queueImpl(hQueue);

    ScopedContext active(hQueue->get_context());

    hQueue->for_each_stream([](hipStream_t s) {
      UR_CHECK_ERROR(hipStreamSynchronize(s));
      UR_CHECK_ERROR(hipStreamDestroy(s));
    });

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // set default result to a negative result (avoid false-positve tests)
  ur_result_t result = UR_RESULT_ERROR_OUT_OF_RESOURCES;

  try {

    ScopedContext active(hQueue->get_context());

    hQueue->sync_streams<true>([&result](hipStream_t s) {
      result = UR_CHECK_ERROR(hipStreamSynchronize(s));
    });

  } catch (ur_result_t err) {

    result = err;

  } catch (...) {

    result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return result;
}

// There is no HIP counterpart for queue flushing and we don't run into the
// same problem of having to flush cross-queue dependencies as some of the
// other plugins, so it can be left as no-op.
UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t hQueue) {
  std::ignore = hQueue;
  return UR_RESULT_SUCCESS;
}

/// Gets the native HIP handle of a UR queue object
///
/// \param[in] queue The UR queue to get the native HIP object of.
/// \param[out] nativeHandle Set to the native handle of the UR queue object.
///
/// \return UR_RESULT_SUCCESS
UR_APIEXPORT ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t hQueue, ur_native_handle_t *phNativeQueue) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeQueue, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ScopedContext active(hQueue->get_context());
  *phNativeQueue =
      reinterpret_cast<ur_native_handle_t>(hQueue->get_next_compute_stream());
  return UR_RESULT_SUCCESS;
}

/// Created a UR queue object from a HIP queue handle.
/// TODO: Implement this.
/// NOTE: The created UR object takes ownership of the native handle.
///
/// \param[in] nativeHandle The native handle to create UR queue object from.
/// \param[in] context is the UR context of the queue.
/// \param[out] queue Set to the UR queue object created from native handle.
/// \param ownNativeHandle tells if SYCL RT should assume the ownership of
///        the native handle, if it can.
///
///
/// \return TBD
UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {

  std::ignore = hNativeQueue;
  std::ignore = hContext;
  std::ignore = hDevice;
  std::ignore = pProperties;
  std::ignore = phQueue;
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
