//===--------- queue.cpp - CUDA Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event.hpp"

#include <cassert>
#include <cuda.h>

void ur_queue_handle_t_::computeStreamWaitForBarrierIfNeeded(CUstream Stream,
                                                             uint32_t StreamI) {
  if (BarrierEvent && !ComputeAppliedBarrier[StreamI]) {
    UR_CHECK_ERROR(cuStreamWaitEvent(Stream, BarrierEvent, 0));
    ComputeAppliedBarrier[StreamI] = true;
  }
}

void ur_queue_handle_t_::transferStreamWaitForBarrierIfNeeded(
    CUstream Stream, uint32_t StreamI) {
  if (BarrierEvent && !TransferAppliedBarrier[StreamI]) {
    UR_CHECK_ERROR(cuStreamWaitEvent(Stream, BarrierEvent, 0));
    TransferAppliedBarrier[StreamI] = true;
  }
}

CUstream ur_queue_handle_t_::getNextComputeStream(uint32_t *StreamToken) {
  uint32_t StreamI;
  uint32_t Token;
  while (true) {
    if (NumComputeStreams < ComputeStreams.size()) {
      // the check above is for performance - so as not to lock mutex every time
      std::lock_guard<std::mutex> guard(ComputeStreamMutex);
      // The second check is done after mutex is locked so other threads can not
      // change NumComputeStreams after that
      if (NumComputeStreams < ComputeStreams.size()) {
        UR_CHECK_ERROR(
            cuStreamCreate(&ComputeStreams[NumComputeStreams++], Flags));
      }
    }
    Token = ComputeStreamIndex++;
    StreamI = Token % ComputeStreams.size();
    // if a stream has been reused before it was next selected round-robin
    // fashion, we want to delay its next use and instead select another one
    // that is more likely to have completed all the enqueued work.
    if (DelayCompute[StreamI]) {
      DelayCompute[StreamI] = false;
    } else {
      break;
    }
  }
  if (StreamToken) {
    *StreamToken = Token;
  }
  CUstream res = ComputeStreams[StreamI];
  computeStreamWaitForBarrierIfNeeded(res, StreamI);
  return res;
}

CUstream ur_queue_handle_t_::getNextComputeStream(
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_stream_guard_ &Guard, uint32_t *StreamToken) {
  for (uint32_t i = 0; i < NumEventsInWaitList; i++) {
    uint32_t Token = EventWaitList[i]->getComputeStreamToken();
    if (reinterpret_cast<ur_queue_handle_t>(EventWaitList[i]->getQueue()) ==
            this &&
        canReuseStream(Token)) {
      std::unique_lock<std::mutex> ComputeSyncGuard(ComputeStreamSyncMutex);
      // redo the check after lock to avoid data races on
      // LastSyncComputeStreams
      if (canReuseStream(Token)) {
        uint32_t StreamI = Token % DelayCompute.size();
        DelayCompute[StreamI] = true;
        if (StreamToken) {
          *StreamToken = Token;
        }
        Guard = ur_stream_guard_{std::move(ComputeSyncGuard)};
        CUstream Result = EventWaitList[i]->getStream();
        computeStreamWaitForBarrierIfNeeded(Result, StreamI);
        return Result;
      }
    }
  }
  Guard = {};
  return getNextComputeStream(StreamToken);
}

CUstream ur_queue_handle_t_::getNextTransferStream() {
  if (TransferStreams.empty()) { // for example in in-order queue
    return getNextComputeStream();
  }
  if (NumTransferStreams < TransferStreams.size()) {
    // the check above is for performance - so as not to lock mutex every time
    std::lock_guard<std::mutex> Guuard(TransferStreamMutex);
    // The second check is done after mutex is locked so other threads can not
    // change NumTransferStreams after that
    if (NumTransferStreams < TransferStreams.size()) {
      UR_CHECK_ERROR(
          cuStreamCreate(&TransferStreams[NumTransferStreams++], Flags));
    }
  }
  uint32_t StreamI = TransferStreamIndex++ % TransferStreams.size();
  CUstream Result = TransferStreams[StreamI];
  transferStreamWaitForBarrierIfNeeded(Result, StreamI);
  return Result;
}

/// Creates a `ur_queue_handle_t` object on the CUDA backend.
/// Valid properties
/// * __SYCL_PI_CUDA_USE_DEFAULT_STREAM -> CU_STREAM_DEFAULT
/// * __SYCL_PI_CUDA_SYNC_WITH_DEFAULT -> CU_STREAM_NON_BLOCKING
UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
              const ur_queue_properties_t *pProps, ur_queue_handle_t *phQueue) {
  try {
    std::unique_ptr<ur_queue_handle_t_> Queue{nullptr};

    if (hContext->getDevice() != hDevice) {
      *phQueue = nullptr;
      return UR_RESULT_ERROR_INVALID_DEVICE;
    }

    unsigned int Flags = CU_STREAM_NON_BLOCKING;
    ur_queue_flags_t URFlags = 0;
    bool IsOutOfOrder = false;
    if (pProps && pProps->stype == UR_STRUCTURE_TYPE_QUEUE_PROPERTIES) {
      URFlags = pProps->flags;
      if (URFlags == UR_QUEUE_FLAG_USE_DEFAULT_STREAM) {
        Flags = CU_STREAM_DEFAULT;
      } else if (URFlags == UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM) {
        Flags = 0;
      }

      if (URFlags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        IsOutOfOrder = true;
      }
    }

    std::vector<CUstream> ComputeCuStreams(
        IsOutOfOrder ? ur_queue_handle_t_::DefaultNumComputeStreams : 1);
    std::vector<CUstream> TransferCuStreams(
        IsOutOfOrder ? ur_queue_handle_t_::DefaultNumTransferStreams : 0);

    Queue = std::unique_ptr<ur_queue_handle_t_>(new ur_queue_handle_t_{
        std::move(ComputeCuStreams), std::move(TransferCuStreams), hContext,
        hDevice, Flags, URFlags});

    *phQueue = Queue.release();

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {

    return Err;

  } catch (...) {

    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRetain(ur_queue_handle_t hQueue) {
  assert(hQueue->getReferenceCount() > 0);

  hQueue->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueRelease(ur_queue_handle_t hQueue) {
  if (hQueue->decrementReferenceCount() > 0) {
    return UR_RESULT_SUCCESS;
  }

  try {
    std::unique_ptr<ur_queue_handle_t_> Queue(hQueue);

    if (!hQueue->backendHasOwnership())
      return UR_RESULT_SUCCESS;

    ScopedContext Active(hQueue->getContext());

    hQueue->forEachStream([](CUstream S) {
      UR_CHECK_ERROR(cuStreamSynchronize(S));
      UR_CHECK_ERROR(cuStreamDestroy(S));
    });

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueFinish(ur_queue_handle_t hQueue) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hQueue->getContext());

    hQueue->syncStreams</*ResetUsed=*/true>(
        [](CUstream s) { UR_CHECK_ERROR(cuStreamSynchronize(s)); });

  } catch (ur_result_t Err) {

    Result = Err;

  } catch (...) {

    Result = UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return Result;
}

// There is no CUDA counterpart for queue flushing and we don't run into the
// same problem of having to flush cross-queue dependencies as some of the
// other plugins, so it can be left as no-op.
UR_APIEXPORT ur_result_t UR_APICALL urQueueFlush(ur_queue_handle_t hQueue) {
  std::ignore = hQueue;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urQueueGetNativeHandle(ur_queue_handle_t hQueue, ur_queue_native_desc_t *pDesc,
                       ur_native_handle_t *phNativeQueue) {
  std::ignore = pDesc;

  ScopedContext Active(hQueue->getContext());
  *phNativeQueue =
      reinterpret_cast<ur_native_handle_t>(hQueue->getNextComputeStream());
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urQueueCreateWithNativeHandle(
    ur_native_handle_t hNativeQueue, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const ur_queue_native_properties_t *pProperties,
    ur_queue_handle_t *phQueue) {
  (void)hDevice;

  unsigned int CuFlags;
  CUstream CuStream = reinterpret_cast<CUstream>(hNativeQueue);

  UR_CHECK_ERROR(cuStreamGetFlags(CuStream, &CuFlags));

  ur_queue_flags_t Flags = 0;
  if (CuFlags == CU_STREAM_DEFAULT)
    Flags = UR_QUEUE_FLAG_USE_DEFAULT_STREAM;
  else if (CuFlags == CU_STREAM_NON_BLOCKING)
    Flags = UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM;
  else
    detail::ur::die("Unknown cuda stream");

  std::vector<CUstream> ComputeCuStreams(1, CuStream);
  std::vector<CUstream> TransferCuStreams(0);

  // Create queue and set num_compute_streams to 1, as computeCuStreams has
  // valid stream
  *phQueue =
      new ur_queue_handle_t_{std::move(ComputeCuStreams),
                             std::move(TransferCuStreams),
                             hContext,
                             hContext->getDevice(),
                             CuFlags,
                             Flags,
                             /*backend_owns*/ pProperties->isNativeHandleOwned};
  (*phQueue)->NumComputeStreams = 1;

  return UR_RESULT_SUCCESS;
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
    try {
      bool IsReady = hQueue->allOf([](CUstream S) -> bool {
        const CUresult Ret = cuStreamQuery(S);
        if (Ret == CUDA_SUCCESS)
          return true;

        if (Ret == CUDA_ERROR_NOT_READY)
          return false;

        UR_CHECK_ERROR(Ret);
        return false;
      });
      return ReturnValue(IsReady);
    } catch (ur_result_t Err) {
      return Err;
    } catch (...) {
      return UR_RESULT_ERROR_OUT_OF_RESOURCES;
    }
  }
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
  case UR_QUEUE_INFO_SIZE:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
}
