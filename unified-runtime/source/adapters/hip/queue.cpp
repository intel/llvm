//===--------- queue.cpp - HIP Adapter ------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "queue.hpp"
#include "context.hpp"
#include "event.hpp"

template <>
void hip_stream_queue::computeStreamWaitForBarrierIfNeeded(hipStream_t Stream,
                                                           uint32_t Stream_i) {
  if (BarrierEvent && !ComputeAppliedBarrier[Stream_i]) {
    UR_CHECK_ERROR(hipStreamWaitEvent(Stream, BarrierEvent, 0));
    ComputeAppliedBarrier[Stream_i] = true;
  }
}

template <>
void hip_stream_queue::transferStreamWaitForBarrierIfNeeded(hipStream_t Stream,
                                                            uint32_t Stream_i) {
  if (BarrierEvent && !TransferAppliedBarrier[Stream_i]) {
    UR_CHECK_ERROR(hipStreamWaitEvent(Stream, BarrierEvent, 0));
    TransferAppliedBarrier[Stream_i] = true;
  }
}

template <>
ur_queue_handle_t hip_stream_queue::getEventQueue(const ur_event_handle_t e) {
  return e->getQueue();
}

template <>
uint32_t
hip_stream_queue::getEventComputeStreamToken(const ur_event_handle_t e) {
  return e->getComputeStreamToken();
}

template <>
hipStream_t hip_stream_queue::getEventStream(const ur_event_handle_t e) {
  return e->getStream();
}

UR_APIEXPORT ur_result_t UR_APICALL
urQueueCreate(ur_context_handle_t hContext, ur_device_handle_t hDevice,
              const ur_queue_properties_t *pProps, ur_queue_handle_t *phQueue) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  try {
    std::unique_ptr<ur_queue_handle_t_> QueueImpl{nullptr};

    unsigned int Flags = hipStreamNonBlocking;
    ur_queue_flags_t URFlags = 0;
    int Priority = 0; // Not guaranteed, but, in ROCm 5.0-6.0, 0 is the default
    if (pProps && pProps->stype == UR_STRUCTURE_TYPE_QUEUE_PROPERTIES) {
      URFlags = pProps->flags;
      if (URFlags == UR_QUEUE_FLAG_USE_DEFAULT_STREAM) {
        Flags = hipStreamDefault;
      } else if (URFlags == UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM) {
        Flags = 0;
      }

      if (URFlags & UR_QUEUE_FLAG_PRIORITY_HIGH) {
        ScopedDevice Active(hDevice);
        UR_CHECK_ERROR(hipDeviceGetStreamPriorityRange(nullptr, &Priority));
      } else if (URFlags & UR_QUEUE_FLAG_PRIORITY_LOW) {
        ScopedDevice Active(hDevice);
        UR_CHECK_ERROR(hipDeviceGetStreamPriorityRange(&Priority, nullptr));
      }
    }

    const bool IsOutOfOrder =
        pProps ? pProps->flags & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE
               : false;

    QueueImpl = std::unique_ptr<ur_queue_handle_t_>(new ur_queue_handle_t_{
        {}, {IsOutOfOrder, hContext, hDevice, Flags, URFlags, Priority}});

    *phQueue = QueueImpl.release();

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {
    return Err;
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
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
  case UR_QUEUE_INFO_DEVICE_DEFAULT:
  case UR_QUEUE_INFO_SIZE:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
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

    ScopedDevice Active(hQueue->getDevice());

    hQueue->forEachStream([](hipStream_t S) {
      UR_CHECK_ERROR(hipStreamSynchronize(S));
      UR_CHECK_ERROR(hipStreamDestroy(S));
    });

    if (hQueue->getHostSubmitTimeStream() != hipStream_t{0}) {
      UR_CHECK_ERROR(hipStreamSynchronize(hQueue->getHostSubmitTimeStream()));
      UR_CHECK_ERROR(hipStreamDestroy(hQueue->getHostSubmitTimeStream()));
    }

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
    ScopedDevice Active(hQueue->getDevice());

    hQueue->syncStreams</*ResetUsed=*/true>(
        [](hipStream_t S) { UR_CHECK_ERROR(hipStreamSynchronize(S)); });

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
  ScopedDevice Active(hQueue->getDevice());
  *phNativeQueue =
      reinterpret_cast<ur_native_handle_t>(hQueue->getInteropStream());
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
  if (!hDevice && hContext->getDevices().size() == 1)
    hDevice = hContext->getDevices().front();

  unsigned int HIPFlags;
  hipStream_t HIPStream = reinterpret_cast<hipStream_t>(hNativeQueue);

  UR_CHECK_ERROR(hipStreamGetFlags(HIPStream, &HIPFlags));

  ur_queue_flags_t Flags = 0;
  if (HIPFlags == hipStreamDefault) {
    Flags = UR_QUEUE_FLAG_USE_DEFAULT_STREAM;
  } else if (HIPFlags == hipStreamNonBlocking) {
    Flags = UR_QUEUE_FLAG_SYNC_WITH_DEFAULT_STREAM;
  } else {
    setErrorMessage("Incorrect native stream flags, expecting "
                    "hipStreamDefault or hipStreamNonBlocking",
                    UR_RESULT_ERROR_INVALID_VALUE);
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  auto isNativeHandleOwned =
      pProperties ? pProperties->isNativeHandleOwned : false;

  // Create queue and set num_compute_streams to 1, as computeHIPStreams has
  // valid stream
  *phQueue = new ur_queue_handle_t_{
      {}, {HIPStream, hContext, hDevice, HIPFlags, Flags, isNativeHandleOwned}};

  return UR_RESULT_SUCCESS;
}
