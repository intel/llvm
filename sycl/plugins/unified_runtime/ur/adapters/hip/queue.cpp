//===--------- queue.cpp - HIP Adapter ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue.hpp"
#include "context.hpp"
#include "event.hpp"

void ur_queue_handle_t_::computeStreamWaitForBarrierIfNeeded(
    hipStream_t Stream, uint32_t Stream_i) {
  if (BarrierEvent && !ComputeAppliedBarrier[Stream_i]) {
    UR_CHECK_ERROR(hipStreamWaitEvent(Stream, BarrierEvent, 0));
    ComputeAppliedBarrier[Stream_i] = true;
  }
}

void ur_queue_handle_t_::transferStreamWaitForBarrierIfNeeded(
    hipStream_t Stream, uint32_t Stream_i) {
  if (BarrierEvent && !TransferAppliedBarrier[Stream_i]) {
    UR_CHECK_ERROR(hipStreamWaitEvent(Stream, BarrierEvent, 0));
    TransferAppliedBarrier[Stream_i] = true;
  }
}

hipStream_t ur_queue_handle_t_::getNextComputeStream(uint32_t *StreamToken) {
  uint32_t Stream_i;
  uint32_t Token;
  while (true) {
    if (NumComputeStreams < ComputeStreams.size()) {
      // the check above is for performance - so as not to lock mutex every time
      std::lock_guard<std::mutex> guard(ComputeStreamMutex);
      // The second check is done after mutex is locked so other threads can not
      // change NumComputeStreams after that
      if (NumComputeStreams < ComputeStreams.size()) {
        UR_CHECK_ERROR(hipStreamCreateWithFlags(
            &ComputeStreams[NumComputeStreams++], Flags));
      }
    }
    Token = ComputeStreamIdx++;
    Stream_i = Token % ComputeStreams.size();
    // if a stream has been reused before it was next selected round-robin
    // fashion, we want to delay its next use and instead select another one
    // that is more likely to have completed all the enqueued work.
    if (DelayCompute[Stream_i]) {
      DelayCompute[Stream_i] = false;
    } else {
      break;
    }
  }
  if (StreamToken) {
    *StreamToken = Token;
  }
  hipStream_t Res = ComputeStreams[Stream_i];
  computeStreamWaitForBarrierIfNeeded(Res, Stream_i);
  return Res;
}

hipStream_t ur_queue_handle_t_::getNextComputeStream(
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_stream_quard &Guard, uint32_t *StreamToken) {
  for (uint32_t i = 0; i < NumEventsInWaitList; i++) {
    uint32_t Token = EventWaitList[i]->getComputeStreamToken();
    if (EventWaitList[i]->getQueue() == this && canReuseStream(Token)) {
      std::unique_lock<std::mutex> ComputeSyncGuard(ComputeStreamSyncMutex);
      // redo the check after lock to avoid data races on
      // LastSyncComputeStreams
      if (canReuseStream(Token)) {
        uint32_t Stream_i = Token % DelayCompute.size();
        DelayCompute[Stream_i] = true;
        if (StreamToken) {
          *StreamToken = Token;
        }
        Guard = ur_stream_quard{std::move(ComputeSyncGuard)};
        hipStream_t Res = EventWaitList[i]->getStream();
        computeStreamWaitForBarrierIfNeeded(Res, Stream_i);
        return Res;
      }
    }
  }
  Guard = {};
  return getNextComputeStream(StreamToken);
}

hipStream_t ur_queue_handle_t_::getNextTransferStream() {
  if (TransferStreams.empty()) { // for example in in-order queue
    return getNextComputeStream();
  }
  if (NumTransferStreams < TransferStreams.size()) {
    // the check above is for performance - so as not to lock mutex every time
    std::lock_guard<std::mutex> Guard(TransferStreamMutex);
    // The second check is done after mutex is locked so other threads can not
    // change NumTransferStreams after that
    if (NumTransferStreams < TransferStreams.size()) {
      UR_CHECK_ERROR(hipStreamCreateWithFlags(
          &TransferStreams[NumTransferStreams++], Flags));
    }
  }
  uint32_t Stream_i = TransferStreamIdx++ % TransferStreams.size();
  hipStream_t Res = TransferStreams[Stream_i];
  transferStreamWaitForBarrierIfNeeded(Res, Stream_i);
  return Res;
}

UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
              const ur_queue_properties_t *pProps, ur_queue_handle_t *phQueue) {
  try {
    std::unique_ptr<ur_queue_handle_t_> QueueImpl{nullptr};

    if (hContext->getDevice() != hDevice) {
      *phQueue = nullptr;
      return UR_RESULT_ERROR_INVALID_DEVICE;
    }

    unsigned int Flags = 0;

    const bool IsOutOfOrder =
        pProps ? pProps->flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE
               : false;

    std::vector<hipStream_t> ComputeHipStreams(
        IsOutOfOrder ? ur_queue_handle_t_::DefaultNumComputeStreams : 1);
    std::vector<hipStream_t> TransferHipStreams(
        IsOutOfOrder ? ur_queue_handle_t_::DefaultNumTransferStreams : 0);

    QueueImpl = std::unique_ptr<ur_queue_handle_t_>(new ur_queue_handle_t_{
        std::move(ComputeHipStreams), std::move(TransferHipStreams), hContext,
        hDevice, Flags, pProps ? pProps->flags : 0});

    *phQueue = QueueImpl.release();

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t hQueue,
                                                   ur_queue_info_t propName,
                                                   size_t propValueSize,
                                                   void *pPropValue,
                                                   size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_QUEUE_INFO_CONTEXT:
    return ReturnValue(hQueue->Context);
  case UR_QUEUE_INFO_DEVICE:
    return ReturnValue(hQueue->Device);
  case UR_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(hQueue->getReferenceCount());
  case UR_QUEUE_INFO_FLAGS:
    return ReturnValue(hQueue->URFlags);
  case UR_QUEUE_INFO_EMPTY: {
    bool IsReady = hQueue->allOf([](hipStream_t S) -> bool {
      const hipError_t Ret = hipStreamQuery(S);
      if (Ret == hipSuccess)
        return true;

      try {
        UR_CHECK_ERROR(Ret);
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
  UR_ASSERT(hQueue->getReferenceCount() > 0, UR_RESULT_ERROR_INVALID_QUEUE);

  hQueue->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  if (hQueue->decrementReferenceCount() > 0) {
    return UR_RESULT_SUCCESS;
  }

  try {
    std::unique_ptr<ur_queue_handle_t_> QueueImpl(hQueue);

    if (!hQueue->backendHasOwnership())
      return UR_RESULT_SUCCESS;

    ScopedContext Active(hQueue->getContext()->getDevice());

    hQueue->forEachStream([](hipStream_t S) {
      UR_CHECK_ERROR(hipStreamSynchronize(S));
      UR_CHECK_ERROR(hipStreamDestroy(S));
    });

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  // set default result to a negative result (avoid false-positve tests)
  ur_result_t Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;

  try {

    ScopedContext Active(hQueue->getContext()->getDevice());

    hQueue->syncStreams<true>([&Result](hipStream_t S) {
      UR_CHECK_ERROR(hipStreamSynchronize(S));
      Result = UR_RESULT_SUCCESS;
    });

  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return Result;
}

// There is no HIP counterpart for queue flushing and we don't run into the
// same problem of having to flush cross-queue dependencies as some of the
// other plugins, so it can be left as no-op.
UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t) {
  return UR_RESULT_SUCCESS;
}

/// Gets the native HIP handle of a UR queue object
///
/// \param[in] hQueue The UR queue to get the native HIP object of.
/// \param[out] phNativeQueue Set to the native handle of the UR queue object.
///
/// \return UR_RESULT_SUCCESS
UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetNativeHandle(ur_queue_handle_t hQueue, ur_queue_native_desc_t *,
                       ur_native_handle_t *phNativeQueue) {
  ScopedContext Active(hQueue->getContext()->getDevice());
  *phNativeQueue =
      reinterpret_cast<ur_native_handle_t>(hQueue->getNextComputeStream());
  return UR_RESULT_SUCCESS;
}

/// Created a UR queue object from a HIP queue handle.
/// NOTE: The created UR object doesn't takes ownership of the native handle.
///
/// \param[in] hNativeQueue The native handle to create UR queue object from.
/// \param[in] hContext is the UR context of the queue.
/// \param[out] phQueue Set to the UR queue object created from native handle.
UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {
  (void)hDevice;

  unsigned int HIPFlags;
  hipStream_t HIPStream = reinterpret_cast<hipStream_t>(hNativeQueue);

  UR_CHECK_ERROR(hipStreamGetFlags(HIPStream, &HIPFlags));

  ur_queue_flags_t Flags = 0;
  if (HIPFlags == hipStreamDefault)
    Flags = UR_QUEUE_FLAG_USE_DEFAULT_STREAM;
  else if (HIPFlags == hipStreamNonBlocking)
    Flags = UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM;
  else
    detail::ur::die("Unknown hip stream");

  std::vector<hipStream_t> ComputeHIPStreams(1, HIPStream);
  std::vector<hipStream_t> TransferHIPStreams(0);

  // Create queue and set num_compute_streams to 1, as computeHIPStreams has
  // valid stream
  *phQueue =
      new ur_queue_handle_t_{std::move(ComputeHIPStreams),
                             std::move(TransferHIPStreams),
                             hContext,
                             hContext->getDevice(),
                             HIPFlags,
                             Flags,
                             /*backend_owns*/ pProperties->isNativeHandleOwned};
  (*phQueue)->NumComputeStreams = 1;

  return UR_RESULT_SUCCESS;
}
