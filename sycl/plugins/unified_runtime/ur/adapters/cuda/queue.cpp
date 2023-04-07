//===--------- queue.cpp - CUDA Adapter ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "queue.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event.hpp"

#include <cassert>
#include <cuda.h>

void ur_queue_handle_t_::compute_stream_wait_for_barrier_if_needed(
    CUstream stream, uint32_t stream_i) {
  if (barrier_event_ && !compute_applied_barrier_[stream_i]) {
    UR_CHECK_ERROR(cuStreamWaitEvent(stream, barrier_event_, 0));
    compute_applied_barrier_[stream_i] = true;
  }
}

void ur_queue_handle_t_::transfer_stream_wait_for_barrier_if_needed(
    CUstream stream, uint32_t stream_i) {
  if (barrier_event_ && !transfer_applied_barrier_[stream_i]) {
    UR_CHECK_ERROR(cuStreamWaitEvent(stream, barrier_event_, 0));
    transfer_applied_barrier_[stream_i] = true;
  }
}

CUstream ur_queue_handle_t_::get_next_compute_stream(uint32_t *stream_token) {
  uint32_t stream_i;
  uint32_t token;
  while (true) {
    if (num_compute_streams_ < compute_streams_.size()) {
      // the check above is for performance - so as not to lock mutex every time
      std::lock_guard<std::mutex> guard(compute_stream_mutex_);
      // The second check is done after mutex is locked so other threads can not
      // change num_compute_streams_ after that
      if (num_compute_streams_ < compute_streams_.size()) {
        UR_CHECK_ERROR(
            cuStreamCreate(&compute_streams_[num_compute_streams_++], flags_));
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
  CUstream res = compute_streams_[stream_i];
  compute_stream_wait_for_barrier_if_needed(res, stream_i);
  return res;
}

CUstream ur_queue_handle_t_::get_next_compute_stream(
    uint32_t num_events_in_wait_list, const ur_event_handle_t *event_wait_list,
    ur_stream_guard_ &guard, uint32_t *stream_token) {
  for (uint32_t i = 0; i < num_events_in_wait_list; i++) {
    uint32_t token = event_wait_list[i]->get_compute_stream_token();
    if (reinterpret_cast<ur_queue_handle_t>(event_wait_list[i]->get_queue()) ==
            this &&
        can_reuse_stream(token)) {
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
        guard = ur_stream_guard_{std::move(compute_sync_guard)};
        CUstream res = event_wait_list[i]->get_stream();
        compute_stream_wait_for_barrier_if_needed(res, stream_i);
        return res;
      }
    }
  }
  guard = {};
  return get_next_compute_stream(stream_token);
}

CUstream ur_queue_handle_t_::get_next_transfer_stream() {
  if (transfer_streams_.empty()) { // for example in in-order queue
    return get_next_compute_stream();
  }
  if (num_transfer_streams_ < transfer_streams_.size()) {
    // the check above is for performance - so as not to lock mutex every time
    std::lock_guard<std::mutex> guard(transfer_stream_mutex_);
    // The second check is done after mutex is locked so other threads can not
    // change num_transfer_streams_ after that
    if (num_transfer_streams_ < transfer_streams_.size()) {
      UR_CHECK_ERROR(
          cuStreamCreate(&transfer_streams_[num_transfer_streams_++], flags_));
    }
  }
  uint32_t stream_i = transfer_stream_idx_++ % transfer_streams_.size();
  CUstream res = transfer_streams_[stream_i];
  transfer_stream_wait_for_barrier_if_needed(res, stream_i);
  return res;
}

/// Creates a `ur_queue_handle_t` object on the CUDA backend.
/// Valid properties
/// * __SYCL_PI_CUDA_USE_DEFAULT_STREAM -> CU_STREAM_DEFAULT
/// * __SYCL_PI_CUDA_SYNC_WITH_DEFAULT -> CU_STREAM_NON_BLOCKING
///
UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
              const ur_queue_properties_t *pProps, ur_queue_handle_t *phQueue) {
  try {
    std::unique_ptr<ur_queue_handle_t_> queueImpl{nullptr};

    if (hContext->get_device() != hDevice) {
      *phQueue = nullptr;
      return UR_RESULT_ERROR_INVALID_DEVICE;
    }

    unsigned int flags = CU_STREAM_NON_BLOCKING;
    ur_queue_flags_t urFlags = 0;
    bool is_out_of_order = false;
    if (pProps && pProps->stype == UR_STRUCTURE_TYPE_QUEUE_PROPERTIES) {
      urFlags = pProps->flags;
      if (urFlags == __SYCL_UR_CUDA_USE_DEFAULT_STREAM) {
        flags = CU_STREAM_DEFAULT;
      } else if (urFlags == __SYCL_UR_CUDA_SYNC_WITH_DEFAULT) {
        flags = 0;
      }

      if (urFlags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        is_out_of_order = true;
      }
    }

    std::vector<CUstream> computeCuStreams(
        is_out_of_order ? ur_queue_handle_t_::default_num_compute_streams : 1);
    std::vector<CUstream> transferCuStreams(
        is_out_of_order ? ur_queue_handle_t_::default_num_transfer_streams : 0);

    queueImpl = std::unique_ptr<ur_queue_handle_t_>(new ur_queue_handle_t_{
        std::move(computeCuStreams), std::move(transferCuStreams), hContext,
        hDevice, flags, urFlags});

    *phQueue = queueImpl.release();

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t err) {

    return err;

  } catch (...) {

    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  assert(hQueue != nullptr);
  assert(hQueue->get_reference_count() > 0);

  hQueue->increment_reference_count();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  assert(hQueue != nullptr);

  if (hQueue->decrement_reference_count() > 0) {
    return UR_RESULT_SUCCESS;
  }

  try {
    std::unique_ptr<ur_queue_handle_t_> queueImpl(hQueue);

    if (!hQueue->backend_has_ownership())
      return UR_RESULT_SUCCESS;

    ScopedContext active(hQueue->get_context());

    hQueue->for_each_stream([](CUstream s) {
      UR_CHECK_ERROR(cuStreamSynchronize(s));
      UR_CHECK_ERROR(cuStreamDestroy(s));
    });

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  ur_result_t result = UR_RESULT_SUCCESS;

  try {

    assert(hQueue !=
           nullptr); // need PI_ERROR_INVALID_EXTERNAL_HANDLE error code
    ScopedContext active(hQueue->get_context());

    hQueue->sync_streams</*ResetUsed=*/true>([&result](CUstream s) {
      result = UR_CHECK_ERROR(cuStreamSynchronize(s));
    });

  } catch (ur_result_t err) {

    result = err;

  } catch (...) {

    result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return result;
}

// There is no CUDA counterpart for queue flushing and we don't run into the
// same problem of having to flush cross-queue dependencies as some of the
// other plugins, so it can be left as no-op.
UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t hQueue) {
  (void)hQueue;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetNativeHandle(
    ur_queue_handle_t hQueue, ur_native_handle_t *phNativeQueue) {
  ScopedContext active(hQueue->get_context());
  *phNativeQueue =
      reinterpret_cast<ur_native_handle_t>(hQueue->get_next_compute_stream());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {
  (void)pProperties;

  unsigned int cuFlags;
  CUstream cuStream = reinterpret_cast<CUstream>(hNativeQueue);
  UR_ASSERT(hContext->get_device() == hDevice, UR_RESULT_ERROR_INVALID_DEVICE);

  auto retErr = UR_CHECK_ERROR(cuStreamGetFlags(cuStream, &cuFlags));

  ur_queue_flags_t flags = 0;
  if (cuFlags == CU_STREAM_DEFAULT)
    flags = __SYCL_UR_CUDA_USE_DEFAULT_STREAM;
  else if (cuFlags == CU_STREAM_NON_BLOCKING)
    flags = __SYCL_UR_CUDA_SYNC_WITH_DEFAULT;
  else
    sycl::detail::ur::die("Unknown cuda stream");

  std::vector<CUstream> computeCuStreams(1, cuStream);
  std::vector<CUstream> transferCuStreams(0);

  // Create queue and set num_compute_streams to 1, as computeCuStreams has
  // valid stream
  *phQueue = new ur_queue_handle_t_{std::move(computeCuStreams),
                                    std::move(transferCuStreams),
                                    hContext,
                                    hDevice,
                                    cuFlags,
                                    flags,
                                    /*backend_owns*/ false};
  (*phQueue)->num_compute_streams_ = 1;

  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_QUEUE);

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropSizeRet);

  switch (uint32_t{propName}) {
  case UR_QUEUE_INFO_CONTEXT:
    return ReturnValue(hQueue->context_);
  case UR_QUEUE_INFO_DEVICE:
    return ReturnValue(hQueue->device_);
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(hQueue->get_reference_count());
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(hQueue->ur_flags_);
  case UR_QUEUE_INFO_EMPTY: {
    try {
      bool IsReady = hQueue->all_of([](CUstream s) -> bool {
        const CUresult ret = cuStreamQuery(s);
        if (ret == CUDA_SUCCESS)
          return true;

        if (ret == CUDA_ERROR_NOT_READY)
          return false;

        UR_CHECK_ERROR(ret);
        return false;
      });
      return ReturnValue(IsReady);
    } catch (ur_result_t err) {
      return err;
    } catch (...) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }
  }
  default:
    break;
  }
  sycl::detail::ur::die("Queue info request not implemented");
  return {};
}
