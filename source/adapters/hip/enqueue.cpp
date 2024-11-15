//===--------- enqueue.cpp - HIP Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "enqueue.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"
#include "ur_api.h"

#include <ur/ur.hpp>

extern size_t imageElementByteSize(hipArray_Format ArrayFormat);

ur_result_t enqueueEventsWait(ur_queue_handle_t Queue, hipStream_t Stream,
                              uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList) {
  if (!EventWaitList) {
    return UR_RESULT_SUCCESS;
  }
  try {
    UR_CHECK_ERROR(forLatestEvents(
        EventWaitList, NumEventsInWaitList,
        [Stream, Queue](ur_event_handle_t Event) -> ur_result_t {
          ScopedDevice Active(Queue->getDevice());
          if (Event->isCompleted() || Event->getStream() == Stream) {
            return UR_RESULT_SUCCESS;
          } else {
            UR_CHECK_ERROR(hipStreamWaitEvent(Stream, Event->get(), 0));
            return UR_RESULT_SUCCESS;
          }
        }));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

// Determine local work sizes that result in uniform work groups.
// The default threadsPerBlock only require handling the first work_dim
// dimension.
void guessLocalWorkSize(ur_device_handle_t Device, size_t *ThreadsPerBlock,
                        const size_t *GlobalWorkSize, const uint32_t WorkDim,
                        const size_t MaxThreadsPerBlock[3]) {
  assert(ThreadsPerBlock != nullptr);
  assert(GlobalWorkSize != nullptr);

  // FIXME: The below assumes a three dimensional range but this is not
  // guaranteed by UR.
  size_t GlobalSizeNormalized[3] = {1, 1, 1};
  for (uint32_t i = 0; i < WorkDim; i++) {
    GlobalSizeNormalized[i] = GlobalWorkSize[i];
  }

  size_t MaxBlockDim[3];
  MaxBlockDim[0] = MaxThreadsPerBlock[0];
  MaxBlockDim[1] = Device->getMaxBlockDimY();
  MaxBlockDim[2] = Device->getMaxBlockDimZ();

  roundToHighestFactorOfGlobalSizeIn3d(ThreadsPerBlock, GlobalSizeNormalized,
                                       MaxBlockDim, MaxThreadsPerBlock[0]);
}

namespace {

ur_result_t setHipMemAdvise(const void *DevPtr, const size_t Size,
                            ur_usm_advice_flags_t URAdviceFlags,
                            hipDevice_t Device) {
  // Handle unmapped memory advice flags
  if (URAdviceFlags &
      (UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY |
       UR_USM_ADVICE_FLAG_CLEAR_NON_ATOMIC_MOSTLY |
       UR_USM_ADVICE_FLAG_BIAS_CACHED | UR_USM_ADVICE_FLAG_BIAS_UNCACHED
#if !defined(__HIP_PLATFORM_AMD__)
       | UR_USM_ADVICE_FLAG_SET_NON_COHERENT_MEMORY |
       UR_USM_ADVICE_FLAG_CLEAR_NON_COHERENT_MEMORY
#endif
       )) {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  using ur_to_hip_advice_t = std::pair<ur_usm_advice_flags_t, hipMemoryAdvise>;

#if defined(__HIP_PLATFORM_AMD__)
  constexpr size_t DeviceFlagCount = 8;
#else
  constexpr size_t DeviceFlagCount = 6;
#endif
  static constexpr std::array<ur_to_hip_advice_t, DeviceFlagCount>
      URToHIPMemAdviseDeviceFlags {
    std::make_pair(UR_USM_ADVICE_FLAG_SET_READ_MOSTLY,
                   hipMemAdviseSetReadMostly),
        std::make_pair(UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY,
                       hipMemAdviseUnsetReadMostly),
        std::make_pair(UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION,
                       hipMemAdviseSetPreferredLocation),
        std::make_pair(UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION,
                       hipMemAdviseUnsetPreferredLocation),
        std::make_pair(UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE,
                       hipMemAdviseSetAccessedBy),
        std::make_pair(UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE,
                       hipMemAdviseUnsetAccessedBy),
#if defined(__HIP_PLATFORM_AMD__)
        std::make_pair(UR_USM_ADVICE_FLAG_SET_NON_COHERENT_MEMORY,
                       hipMemAdviseSetCoarseGrain),
        std::make_pair(UR_USM_ADVICE_FLAG_CLEAR_NON_COHERENT_MEMORY,
                       hipMemAdviseUnsetCoarseGrain),
#endif
  };
  for (const auto &[URAdvice, HIPAdvice] : URToHIPMemAdviseDeviceFlags) {
    if (URAdviceFlags & URAdvice) {
      UR_CHECK_ERROR(hipMemAdvise(DevPtr, Size, HIPAdvice, Device));
    }
  }

  static constexpr std::array<ur_to_hip_advice_t, 4> URToHIPMemAdviseHostFlags{
      std::make_pair(UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION_HOST,
                     hipMemAdviseSetPreferredLocation),
      std::make_pair(UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION_HOST,
                     hipMemAdviseUnsetPreferredLocation),
      std::make_pair(UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST,
                     hipMemAdviseSetAccessedBy),
      std::make_pair(UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST,
                     hipMemAdviseUnsetAccessedBy),
  };

  for (const auto &[URAdvice, HIPAdvice] : URToHIPMemAdviseHostFlags) {
    if (URAdviceFlags & URAdvice) {
      UR_CHECK_ERROR(hipMemAdvise(DevPtr, Size, HIPAdvice, hipCpuDeviceId));
    }
  }

  return UR_RESULT_SUCCESS;
}

} // namespace

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
  hBuffer->setLastQueueWritingToMemObj(hQueue);

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_WRITE, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(
        hipMemcpyHtoDAsync(std::get<BufferMem>(hBuffer->Mem)
                               .getPtrWithOffset(hQueue->getDevice(), offset),
                           const_cast<void *>(pSrc), size, HIPStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingWrite) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    // Note that this entry point may be called on a queue that may not be the
    // last queue to write to the MemBuffer, meaning we must perform the copy
    // from a different device
    if (hBuffer->LastQueueWritingToMemObj &&
        hBuffer->LastQueueWritingToMemObj->getDevice() != hQueue->getDevice()) {
      hQueue = hBuffer->LastQueueWritingToMemObj;
    }

    auto Device = hQueue->getDevice();
    ScopedDevice Active(Device);
    hipStream_t HIPStream = hQueue->getNextTransferStream();

    // Use the default stream if copying from another device
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    // Copying from the device with latest version of memory, not necessarily
    // the device associated with the Queue
    UR_CHECK_ERROR(hipMemcpyDtoHAsync(
        pDst,
        std::get<BufferMem>(hBuffer->Mem).getPtrWithOffset(Device, offset),
        size, HIPStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingRead) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t err) {
    return err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(hQueue->getContext() == hKernel->getContext(),
            UR_RESULT_ERROR_INVALID_QUEUE);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  // Early exit for zero size range kernel
  if (*pGlobalWorkSize == 0) {
    return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                          phEventWaitList, phEvent);
  }

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  size_t ThreadsPerBlock[3] = {32u, 1u, 1u};
  size_t BlocksPerGrid[3] = {1u, 1u, 1u};

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ur_device_handle_t Dev = hQueue->getDevice();

    hipFunction_t HIPFunc = hKernel->get();
    UR_CHECK_ERROR(setKernelParams(Dev, workDim, pGlobalWorkOffset,
                                   pGlobalWorkSize, pLocalWorkSize, hKernel,
                                   HIPFunc, ThreadsPerBlock, BlocksPerGrid));

    ScopedDevice Active(Dev);

    uint32_t StreamToken;
    ur_stream_guard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    // For memory migration across devices in the same context
    if (hQueue->getContext()->Devices.size() > 1) {
      for (auto &MemArg : hKernel->Args.MemObjArgs) {
        enqueueMigrateMemoryToDeviceIfNeeded(MemArg.Mem, hQueue->getDevice(),
                                             HIPStream);
        if (MemArg.AccessFlags &
            (UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_WRITE_ONLY)) {
          MemArg.Mem->setLastQueueWritingToMemObj(hQueue);
        }
      }
    }

    auto ArgIndices = hKernel->getArgIndices();

    // If migration of mem across buffer is needed, an event must be associated
    // with this command, implicitly if phEvent is nullptr
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_KERNEL_LAUNCH, hQueue, HIPStream, StreamToken));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(hipModuleLaunchKernel(
        HIPFunc, BlocksPerGrid[0], BlocksPerGrid[1], BlocksPerGrid[2],
        ThreadsPerBlock[0], ThreadsPerBlock[1], ThreadsPerBlock[2],
        hKernel->getLocalSize(), HIPStream, ArgIndices.data(), nullptr));

    hKernel->clearLocalSize();

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t err) {
    return err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueCooperativeKernelLaunchExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return urEnqueueKernelLaunch(hQueue, hKernel, workDim, pGlobalWorkOffset,
                               pGlobalWorkSize, pLocalWorkSize,
                               numEventsInWaitList, phEventWaitList, phEvent);
}

/// Enqueues a wait on the given queue for all events.
/// See \ref enqueueEventWait
///
/// Currently queues are represented by a single in-order stream, therefore
/// every command is an implicit barrier and so urEnqueueEventWait has the
/// same behavior as urEnqueueEventWaitWithBarrier. So urEnqueueEventWait can
/// just call urEnqueueEventWaitWithBarrier.
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}

/// Enqueues a wait on the given queue for all specified events.
/// See \ref enqueueEventWaitWithBarrier
///
/// If the events list is empty, the enqueued wait will wait on all previous
/// events in the queue.
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST)
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST)

  try {
    ScopedDevice Active(hQueue->getDevice());
    uint32_t StreamToken;
    ur_stream_guard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList,
        reinterpret_cast<const ur_event_handle_t *>(phEventWaitList), Guard,
        &StreamToken);
    {
      std::lock_guard<std::mutex> Guard(hQueue->BarrierMutex);
      if (hQueue->BarrierEvent == nullptr) {
        UR_CHECK_ERROR(hipEventCreate(&hQueue->BarrierEvent));
      }
      if (numEventsInWaitList == 0) { //  wait on all work
        if (hQueue->BarrierTmpEvent == nullptr) {
          UR_CHECK_ERROR(hipEventCreate(&hQueue->BarrierTmpEvent));
        }
        hQueue->syncStreams(
            [HIPStream, TmpEvent = hQueue->BarrierTmpEvent](hipStream_t S) {
              if (HIPStream != S) {
                UR_CHECK_ERROR(hipEventRecord(TmpEvent, S));
                UR_CHECK_ERROR(hipStreamWaitEvent(HIPStream, TmpEvent, 0));
              }
            });
      } else { // wait just on given events
        forLatestEvents(
            reinterpret_cast<const ur_event_handle_t *>(phEventWaitList),
            numEventsInWaitList,
            [HIPStream](ur_event_handle_t Event) -> ur_result_t {
              if (Event->getQueue()->hasBeenSynchronized(
                      Event->getComputeStreamToken())) {
                return UR_RESULT_SUCCESS;
              } else {
                UR_CHECK_ERROR(hipStreamWaitEvent(HIPStream, Event->get(), 0));
                return UR_RESULT_SUCCESS;
              }
            });
      }

      UR_CHECK_ERROR(hipEventRecord(hQueue->BarrierEvent, HIPStream));
      for (unsigned int i = 0; i < hQueue->ComputeAppliedBarrier.size(); i++) {
        hQueue->ComputeAppliedBarrier[i] = false;
      }
      for (unsigned int i = 0; i < hQueue->TransferAppliedBarrier.size(); i++) {
        hQueue->TransferAppliedBarrier[i] = false;
      }
    }

    if (phEvent) {
      *phEvent = ur_event_handle_t_::makeNative(
          UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, hQueue, HIPStream, StreamToken);
      UR_CHECK_ERROR((*phEvent)->start());
      UR_CHECK_ERROR((*phEvent)->record());
    }

    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

UR_APIEXPORT ur_result_t urEnqueueEventsWaitWithBarrierExt(
    ur_queue_handle_t hQueue, const ur_exp_enqueue_ext_properties_t *,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}

/// General 3D memory copy operation.
/// This function requires the corresponding HIP context to be at the top of
/// the context stack
/// If the source and/or destination is on the device, SrcPtr and/or DstPtr
/// must be a pointer to a hipDevPtr
static ur_result_t commonEnqueueMemBufferCopyRect(
    hipStream_t HipStream, ur_rect_region_t Region, const void *SrcPtr,
    const hipMemoryType SrcType, ur_rect_offset_t SrcOffset, size_t SrcRowPitch,
    size_t SrcSlicePitch, void *DstPtr, const hipMemoryType DstType,
    ur_rect_offset_t DstOffset, size_t DstRowPitch, size_t DstSlicePitch) {

  assert(SrcType == hipMemoryTypeDevice || SrcType == hipMemoryTypeHost);
  assert(DstType == hipMemoryTypeDevice || DstType == hipMemoryTypeHost);

  SrcRowPitch = (!SrcRowPitch) ? Region.width : SrcRowPitch;
  SrcSlicePitch =
      (!SrcSlicePitch) ? (Region.height * SrcRowPitch) : SrcSlicePitch;
  DstRowPitch = (!DstRowPitch) ? Region.width : DstRowPitch;
  DstSlicePitch =
      (!DstSlicePitch) ? (Region.height * DstRowPitch) : DstSlicePitch;

  HIP_MEMCPY3D Params;

  Params.WidthInBytes = Region.width;
  Params.Height = Region.height;
  Params.Depth = Region.depth;

  Params.srcMemoryType = SrcType;
  Params.srcDevice = SrcType == hipMemoryTypeDevice
                         ? *static_cast<const hipDeviceptr_t *>(SrcPtr)
                         : 0;
  Params.srcHost = SrcType == hipMemoryTypeHost ? SrcPtr : nullptr;
  Params.srcXInBytes = SrcOffset.x;
  Params.srcY = SrcOffset.y;
  Params.srcZ = SrcOffset.z;
  Params.srcPitch = SrcRowPitch;
  Params.srcHeight = SrcSlicePitch / SrcRowPitch;

  Params.dstMemoryType = DstType;
  Params.dstDevice = DstType == hipMemoryTypeDevice
                         ? *reinterpret_cast<hipDeviceptr_t *>(DstPtr)
                         : 0;
  Params.dstHost = DstType == hipMemoryTypeHost ? DstPtr : nullptr;
  Params.dstXInBytes = DstOffset.x;
  Params.dstY = DstOffset.y;
  Params.dstZ = DstOffset.z;
  Params.dstPitch = DstRowPitch;
  Params.dstHeight = DstSlicePitch / DstRowPitch;

  UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&Params, HipStream));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(!(region.width == 0 || region.height == 0 || region.width == 0),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(!(bufferRowPitch != 0 && bufferRowPitch < region.width),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(!(hostRowPitch != 0 && hostRowPitch < region.width),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(!(bufferSlicePitch != 0 &&
              bufferSlicePitch < region.height * bufferRowPitch),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(!(bufferSlicePitch != 0 && bufferSlicePitch % bufferRowPitch != 0),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(
      !(hostSlicePitch != 0 && hostSlicePitch < region.height * hostRowPitch),
      UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(!(hostSlicePitch != 0 && hostSlicePitch % hostRowPitch != 0),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    // Note that this entry point may be called on a queue that may not be the
    // last queue to write to the MemBuffer, meaning we must perform the copy
    // from a different device
    if (hBuffer->LastQueueWritingToMemObj &&
        hBuffer->LastQueueWritingToMemObj->getDevice() != hQueue->getDevice()) {
      hQueue = hBuffer->LastQueueWritingToMemObj;
    }

    auto Device = hQueue->getDevice();
    ScopedDevice Active(Device);
    hipStream_t HIPStream = hQueue->getNextTransferStream();

    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    void *DevPtr = std::get<BufferMem>(hBuffer->Mem).getVoid(Device);
    UR_CHECK_ERROR(commonEnqueueMemBufferCopyRect(
        HIPStream, region, &DevPtr, hipMemoryTypeDevice, bufferOrigin,
        bufferRowPitch, bufferSlicePitch, pDst, hipMemoryTypeHost, hostOrigin,
        hostRowPitch, hostSlicePitch));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingRead) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  void *DevPtr = std::get<BufferMem>(hBuffer->Mem).getVoid(hQueue->getDevice());
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
  hBuffer->setLastQueueWritingToMemObj(hQueue);

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_WRITE, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(commonEnqueueMemBufferCopyRect(
        HIPStream, region, pSrc, hipMemoryTypeHost, hostOrigin, hostRowPitch,
        hostSlicePitch, &DevPtr, hipMemoryTypeDevice, bufferOrigin,
        bufferRowPitch, bufferSlicePitch));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingWrite) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(size + srcOffset <= std::get<BufferMem>(hBufferSrc->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(size + dstOffset <= std::get<BufferMem>(hBufferDst->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    auto Stream = hQueue->getNextTransferStream();

    if (phEventWaitList) {
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, Stream, numEventsInWaitList,
                                       phEventWaitList));
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY, hQueue, Stream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    auto Src = std::get<BufferMem>(hBufferSrc->Mem)
                   .getPtrWithOffset(hQueue->getDevice(), srcOffset);
    auto Dst = std::get<BufferMem>(hBufferDst->Mem)
                   .getPtrWithOffset(hQueue->getDevice(), dstOffset);

    UR_CHECK_ERROR(hipMemcpyDtoDAsync(Dst, Src, size, Stream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  void *SrcPtr =
      std::get<BufferMem>(hBufferSrc->Mem).getVoid(hQueue->getDevice());
  void *DstPtr =
      std::get<BufferMem>(hBufferDst->Mem).getVoid(hQueue->getDevice());
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(commonEnqueueMemBufferCopyRect(
        HIPStream, region, &SrcPtr, hipMemoryTypeDevice, srcOrigin, srcRowPitch,
        srcSlicePitch, &DstPtr, hipMemoryTypeDevice, dstOrigin, dstRowPitch,
        dstSlicePitch));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}

static inline void memsetRemainPattern(hipStream_t Stream, uint32_t PatternSize,
                                       size_t Size, const void *pPattern,
                                       hipDeviceptr_t Ptr,
                                       uint32_t StartOffset) {
  // Calculate the number of times the pattern needs to be applied
  auto Height = Size / PatternSize;

  for (auto step = StartOffset; step < PatternSize; ++step) {
    // take 1 byte of the pattern
    auto Value = *(static_cast<const uint8_t *>(pPattern) + step);

    // offset the pointer to the part of the buffer we want to write to
    auto OffsetPtr =
        reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(Ptr) + step);

    // set all of the pattern chunks
    UR_CHECK_ERROR(
        hipMemset2DAsync(OffsetPtr, PatternSize, Value, 1u, Height, Stream));
  }
}

// HIP has no memset functions that allow setting values more than 4 bytes. UR
// API lets you pass an arbitrary "pattern" to the buffer fill, which can be
// more than 4 bytes. We must break up the pattern into 1 byte values, and set
// the buffer using multiple strided calls.  The first 4 patterns are set
// using hipMemsetD32Async then all subsequent 1 byte patterns are set using
// hipMemset2DAsync which is called for each pattern.
ur_result_t commonMemSetLargePattern(hipStream_t Stream, uint32_t PatternSize,
                                     size_t Size, const void *pPattern,
                                     hipDeviceptr_t Ptr) {
  // Find the largest supported word size into which the pattern can be divided
  auto BackendWordSize = PatternSize % 4u == 0u   ? 4u
                         : PatternSize % 2u == 0u ? 2u
                                                  : 1u;

  // Calculate the number of patterns
  auto NumberOfSteps = PatternSize / BackendWordSize;

  // If the pattern is 1 word or the first word is repeated throughout, a fast
  // continuous fill can be used without the need for slower strided fills
  bool UseOnlyFirstValue{true};
  auto checkIfFirstWordRepeats = [&UseOnlyFirstValue,
                                  NumberOfSteps](const auto *pPatternWords) {
    for (auto Step{1u}; (Step < NumberOfSteps) && UseOnlyFirstValue; ++Step) {
      if (*(pPatternWords + Step) != *pPatternWords) {
        UseOnlyFirstValue = false;
      }
    }
  };

  // Use a continuous fill for the first word in the pattern because it's faster
  // than a strided fill. Then, overwrite the other values in subsequent steps.
  switch (BackendWordSize) {
  case 4u: {
    auto *pPatternWords = static_cast<const uint32_t *>(pPattern);
    checkIfFirstWordRepeats(pPatternWords);
    UR_CHECK_ERROR(
        hipMemsetD32Async(Ptr, *pPatternWords, Size / BackendWordSize, Stream));
    break;
  }
  case 2u: {
    auto *pPatternWords = static_cast<const uint16_t *>(pPattern);
    checkIfFirstWordRepeats(pPatternWords);
    UR_CHECK_ERROR(
        hipMemsetD16Async(Ptr, *pPatternWords, Size / BackendWordSize, Stream));
    break;
  }
  default: {
    auto *pPatternWords = static_cast<const uint8_t *>(pPattern);
    checkIfFirstWordRepeats(pPatternWords);
    UR_CHECK_ERROR(
        hipMemsetD8Async(Ptr, *pPatternWords, Size / BackendWordSize, Stream));
    break;
  }
  }

  if (UseOnlyFirstValue) {
    return UR_RESULT_SUCCESS;
  }

  // There is a bug in ROCm prior to 6.0.0 version which causes hipMemset2D
  // to behave incorrectly when acting on host pinned memory.
  // In such a case, the memset operation is partially emulated with memcpy.
#if HIP_VERSION_MAJOR < 6
  hipPointerAttribute_t ptrAttribs{};
  UR_CHECK_ERROR(hipPointerGetAttributes(&ptrAttribs, (const void *)Ptr));

  // The hostPointer attribute is non-null also for shared memory allocations.
  // To make sure that this workaround only executes for host pinned memory,
  // we need to check that isManaged attribute is false.
  if (ptrAttribs.hostPointer && !ptrAttribs.isManaged) {
    const auto NumOfCopySteps = Size / PatternSize;
    const auto Offset = BackendWordSize;
    const auto LeftPatternSize = PatternSize - Offset;
    const auto OffsetPatternPtr = reinterpret_cast<const void *>(
        reinterpret_cast<const uint8_t *>(pPattern) + Offset);

    // Loop through the memory area to memset, advancing each time by the
    // PatternSize and memcpy the left over pattern bits.
    for (uint32_t i = 0; i < NumOfCopySteps; ++i) {
      auto OffsetDstPtr = reinterpret_cast<void *>(
          reinterpret_cast<uint8_t *>(Ptr) + Offset + i * PatternSize);
      UR_CHECK_ERROR(hipMemcpyAsync(OffsetDstPtr, OffsetPatternPtr,
                                    LeftPatternSize, hipMemcpyHostToHost,
                                    Stream));
    }
  } else {
    memsetRemainPattern(Stream, PatternSize, Size, pPattern, Ptr,
                        BackendWordSize);
  }
#else
  memsetRemainPattern(Stream, PatternSize, Size, pPattern, Ptr,
                      BackendWordSize);
#endif
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, const void *pPattern,
    size_t patternSize, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(size + offset <= std::get<BufferMem>(hBuffer->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
  hBuffer->setLastQueueWritingToMemObj(hQueue);

  try {
    ScopedDevice Active(hQueue->getDevice());

    auto Stream = hQueue->getNextTransferStream();
    if (phEventWaitList) {
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, Stream, numEventsInWaitList,
                                       phEventWaitList));
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_WRITE, hQueue, Stream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    auto DstDevice = std::get<BufferMem>(hBuffer->Mem)
                         .getPtrWithOffset(hQueue->getDevice(), offset);
    auto N = size / patternSize;

    // pattern size in bytes
    switch (patternSize) {
    case 1: {
      auto Value = *static_cast<const uint8_t *>(pPattern);
      UR_CHECK_ERROR(hipMemsetD8Async(DstDevice, Value, N, Stream));
      break;
    }
    case 2: {
      auto Value = *static_cast<const uint16_t *>(pPattern);
      UR_CHECK_ERROR(hipMemsetD16Async(DstDevice, Value, N, Stream));
      break;
    }
    case 4: {
      auto Value = *static_cast<const uint32_t *>(pPattern);
      UR_CHECK_ERROR(hipMemsetD32Async(DstDevice, Value, N, Stream));
      break;
    }

    default: {
      UR_CHECK_ERROR(commonMemSetLargePattern(Stream, patternSize, size,
                                              pPattern, DstDevice));
      break;
    }
    }

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

/// General ND memory copy operation for images (where N > 1).
/// This function requires the corresponding HIP context to be at the top of
/// the context stack
/// If the source and/or destination is an array, SrcPtr and/or DstPtr
/// must be a pointer to a hipArray
static ur_result_t commonEnqueueMemImageNDCopy(
    hipStream_t HipStream, ur_mem_type_t ImgType, const size_t *Region,
    const void *SrcPtr, const hipMemoryType SrcType, const size_t *SrcOffset,
    void *DstPtr, const hipMemoryType DstType, const size_t *DstOffset) {
  UR_ASSERT(SrcType == hipMemoryTypeArray || SrcType == hipMemoryTypeHost,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(DstType == hipMemoryTypeArray || DstType == hipMemoryTypeHost,
            UR_RESULT_ERROR_INVALID_VALUE);

  if (ImgType == UR_MEM_TYPE_IMAGE1D || ImgType == UR_MEM_TYPE_IMAGE2D) {
    hip_Memcpy2D CpyDesc;
    memset(&CpyDesc, 0, sizeof(CpyDesc));
    CpyDesc.srcMemoryType = SrcType;
    if (SrcType == hipMemoryTypeArray) {
      CpyDesc.srcArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(SrcPtr));
      CpyDesc.srcXInBytes = SrcOffset[0];
      CpyDesc.srcY = (ImgType == UR_MEM_TYPE_IMAGE1D) ? 0 : SrcOffset[1];
    } else {
      CpyDesc.srcHost = SrcPtr;
    }
    CpyDesc.dstMemoryType = DstType;
    if (DstType == hipMemoryTypeArray) {
      CpyDesc.dstArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(DstPtr));
      CpyDesc.dstXInBytes = DstOffset[0];
      CpyDesc.dstY = (ImgType == UR_MEM_TYPE_IMAGE1D) ? 0 : DstOffset[1];
    } else {
      CpyDesc.dstHost = DstPtr;
    }
    CpyDesc.WidthInBytes = Region[0];
    CpyDesc.Height = (ImgType == UR_MEM_TYPE_IMAGE1D) ? 1 : Region[1];
    UR_CHECK_ERROR(hipMemcpyParam2DAsync(&CpyDesc, HipStream));
    return UR_RESULT_SUCCESS;
  }

  if (ImgType == UR_MEM_TYPE_IMAGE3D) {

    HIP_MEMCPY3D CpyDesc;
    memset(&CpyDesc, 0, sizeof(CpyDesc));
    CpyDesc.srcMemoryType = SrcType;
    if (SrcType == hipMemoryTypeArray) {
      CpyDesc.srcArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(SrcPtr));
      CpyDesc.srcXInBytes = SrcOffset[0];
      CpyDesc.srcY = SrcOffset[1];
      CpyDesc.srcZ = SrcOffset[2];
    } else {
      CpyDesc.srcHost = SrcPtr;
    }
    CpyDesc.dstMemoryType = DstType;
    if (DstType == hipMemoryTypeArray) {
      CpyDesc.dstArray = reinterpret_cast<hipCUarray>(DstPtr);
      CpyDesc.dstXInBytes = DstOffset[0];
      CpyDesc.dstY = DstOffset[1];
      CpyDesc.dstZ = DstOffset[2];
    } else {
      CpyDesc.dstHost = DstPtr;
    }
    CpyDesc.WidthInBytes = Region[0];
    CpyDesc.Height = Region[1];
    CpyDesc.Depth = Region[2];
    UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&CpyDesc, HipStream));
    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_ERROR_INVALID_VALUE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingRead,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t, size_t,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(hImage->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  try {
    // Note that this entry point may be called on a queue that may not be the
    // last queue to write to the MemImage, meaning we must perform the copy
    // from a different device
    if (hImage->LastQueueWritingToMemObj &&
        hImage->LastQueueWritingToMemObj->getDevice() != hQueue->getDevice()) {
      hQueue = hImage->LastQueueWritingToMemObj;
    }

    auto Device = hQueue->getDevice();
    ScopedDevice Active(Device);
    hipStream_t HIPStream = hQueue->getNextTransferStream();

    if (phEventWaitList) {
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                       phEventWaitList));
    }

    hipArray *Array = std::get<SurfaceMem>(hImage->Mem).getArray(Device);

    hipArray_Format Format{};
    size_t NumChannels{};
    UR_CHECK_ERROR(getArrayDesc(Array, Format, NumChannels));

    int ElementByteSize = imageElementByteSize(Format);

    size_t ByteOffsetX = origin.x * ElementByteSize * NumChannels;
    size_t BytesToCopy = ElementByteSize * NumChannels * region.width;

    auto ImgType = std::get<SurfaceMem>(hImage->Mem).getImageType();

    size_t AdjustedRegion[3] = {BytesToCopy, region.height, region.depth};
    size_t SrcOffset[3] = {ByteOffsetX, origin.y, origin.z};

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(commonEnqueueMemImageNDCopy(
        HIPStream, ImgType, AdjustedRegion, Array, hipMemoryTypeArray,
        SrcOffset, pDst, hipMemoryTypeHost, nullptr));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

    if (blockingRead) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t, size_t,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(hImage->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();

    if (phEventWaitList) {
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                       phEventWaitList));
    }

    hipArray *Array =
        std::get<SurfaceMem>(hImage->Mem).getArray(hQueue->getDevice());

    hipArray_Format Format{};
    size_t NumChannels{};
    UR_CHECK_ERROR(getArrayDesc(Array, Format, NumChannels));

    int ElementByteSize = imageElementByteSize(Format);

    size_t ByteOffsetX = origin.x * ElementByteSize * NumChannels;
    size_t BytesToCopy = ElementByteSize * NumChannels * region.width;

    auto ImgType = std::get<SurfaceMem>(hImage->Mem).getImageType();

    size_t AdjustedRegion[3] = {BytesToCopy, region.height, region.depth};
    size_t DstOffset[3] = {ByteOffsetX, origin.y, origin.z};

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(commonEnqueueMemImageNDCopy(
        HIPStream, ImgType, AdjustedRegion, pSrc, hipMemoryTypeHost, nullptr,
        Array, hipMemoryTypeArray, DstOffset));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImageSrc,
    ur_mem_handle_t hImageDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hImageSrc->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hImageDst->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(std::get<SurfaceMem>(hImageSrc->Mem).getImageType() ==
                std::get<SurfaceMem>(hImageDst->Mem).getImageType(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    if (phEventWaitList) {
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                       phEventWaitList));
    }

    hipArray *SrcArray =
        std::get<SurfaceMem>(hImageSrc->Mem).getArray(hQueue->getDevice());
    hipArray_Format SrcFormat{};
    size_t SrcNumChannels{};
    UR_CHECK_ERROR(getArrayDesc(SrcArray, SrcFormat, SrcNumChannels));

    hipArray *DstArray =
        std::get<SurfaceMem>(hImageDst->Mem).getArray(hQueue->getDevice());
    hipArray_Format DstFormat{};
    size_t DstNumChannels{};
    UR_CHECK_ERROR(getArrayDesc(DstArray, DstFormat, DstNumChannels));

    UR_ASSERT(SrcFormat == DstFormat,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    UR_ASSERT(SrcNumChannels == DstNumChannels,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

    int ElementByteSize = imageElementByteSize(SrcFormat);

    size_t DstByteOffsetX = dstOrigin.x * ElementByteSize * DstNumChannels;
    size_t SrcByteOffsetX = srcOrigin.x * ElementByteSize * SrcNumChannels;
    size_t BytesToCopy = ElementByteSize * SrcNumChannels * region.width;

    auto ImgType = std::get<SurfaceMem>(hImageSrc->Mem).getImageType();

    size_t AdjustedRegion[3] = {BytesToCopy, region.height, region.depth};
    size_t SrcOffset[3] = {SrcByteOffsetX, srcOrigin.y, srcOrigin.z};
    size_t DstOffset[3] = {DstByteOffsetX, dstOrigin.y, dstOrigin.z};

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(commonEnqueueMemImageNDCopy(
        HIPStream, ImgType, AdjustedRegion, SrcArray, hipMemoryTypeArray,
        SrcOffset, DstArray, hipMemoryTypeArray, DstOffset));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

/// Implements mapping on the host using a BufferRead operation.
/// Mapped pointers are stored in the ur_mem_handle_t object.
/// If the buffer uses pinned host memory a pointer to that memory is returned
/// and no read operation is done.
///
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingMap,
    ur_map_flags_t mapFlags, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, void **ppRetMap) {
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  auto &BufferImpl = std::get<BufferMem>(hBuffer->Mem);
  UR_ASSERT(offset + size <= BufferImpl.getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  auto MapPtr = BufferImpl.mapToPtr(size, offset, mapFlags);
  if (!MapPtr) {
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  }

  const bool IsPinned =
      BufferImpl.MemAllocMode == BufferMem::AllocMode::AllocHostPtr;

  try {
    if (!IsPinned && (mapFlags & (UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE))) {
      // Pinned host memory is already on host so it doesn't need to be read.
      UR_CHECK_ERROR(urEnqueueMemBufferRead(
          hQueue, hBuffer, blockingMap, offset, size, MapPtr,
          numEventsInWaitList, phEventWaitList, phEvent));
    } else {
      ScopedDevice Active(hQueue->getDevice());

      if (IsPinned) {
        UR_CHECK_ERROR(urEnqueueEventsWait(hQueue, numEventsInWaitList,
                                           phEventWaitList, nullptr));
      }

      if (phEvent) {
        *phEvent = ur_event_handle_t_::makeNative(
            UR_COMMAND_MEM_BUFFER_MAP, hQueue, hQueue->getNextTransferStream());
        UR_CHECK_ERROR((*phEvent)->start());
        UR_CHECK_ERROR((*phEvent)->record());
      }
    }
  } catch (ur_result_t Error) {
    return Error;
  }

  *ppRetMap = MapPtr;

  return UR_RESULT_SUCCESS;
}

/// Implements the unmap from the host, using a BufferWrite operation.
/// Requires the mapped pointer to be already registered in the given hMem.
/// If hMem uses pinned host memory, this will not do a write.
///
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hMem->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  auto &BufferImpl = std::get<BufferMem>(hMem->Mem);

  auto *Map = BufferImpl.getMapDetails(pMappedPtr);
  UR_ASSERT(Map != nullptr, UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  const bool IsPinned =
      BufferImpl.MemAllocMode == BufferMem::AllocMode::AllocHostPtr;

  try {
    if (!IsPinned &&
        (Map->getMapFlags() &
         (UR_MAP_FLAG_WRITE | UR_MAP_FLAG_WRITE_INVALIDATE_REGION))) {
      // Pinned host memory is only on host so it doesn't need to be written
      // to.
      UR_CHECK_ERROR(urEnqueueMemBufferWrite(
          hQueue, hMem, true, Map->getMapOffset(), Map->getMapSize(),
          pMappedPtr, numEventsInWaitList, phEventWaitList, phEvent));
    } else {
      ScopedDevice Active(hQueue->getDevice());

      if (IsPinned) {
        UR_CHECK_ERROR(urEnqueueEventsWait(hQueue, numEventsInWaitList,
                                           phEventWaitList, nullptr));
      }

      if (phEvent) {
        *phEvent = ur_event_handle_t_::makeNative(
            UR_COMMAND_MEM_UNMAP, hQueue, hQueue->getNextTransferStream());
        UR_CHECK_ERROR((*phEvent)->start());
        UR_CHECK_ERROR((*phEvent)->record());
      }
    }
  } catch (ur_result_t Error) {
    return Error;
  }

  BufferImpl.unmap(pMappedPtr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    uint32_t StreamToken;
    ur_stream_guard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_FILL, hQueue, HIPStream, StreamToken));
      UR_CHECK_ERROR(EventPtr->start());
    }

    auto N = size / patternSize;
    switch (patternSize) {
    case 1:
      UR_CHECK_ERROR(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(ptr),
                                      *(const uint8_t *)pPattern & 0xFF, N,
                                      HIPStream));
      break;
    case 2:
      UR_CHECK_ERROR(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(ptr),
                                       *(const uint16_t *)pPattern & 0xFFFF, N,
                                       HIPStream));
      break;
    case 4:
      UR_CHECK_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(ptr),
                                       *(const uint32_t *)pPattern & 0xFFFFFFFF,
                                       N, HIPStream));
      break;

    default:
      UR_CHECK_ERROR(
          commonMemSetLargePattern(HIPStream, patternSize, size, pPattern,
                                   reinterpret_cast<hipDeviceptr_t>(ptr)));
      break;
    }

    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
      *phEvent = EventPtr.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, const void *pSrc,
    size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_MEMCPY, hQueue, HIPStream));
      UR_CHECK_ERROR(EventPtr->start());
    }
    UR_CHECK_ERROR(
        hipMemcpyAsync(pDst, pSrc, size, hipMemcpyDefault, HIPStream));
    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
    }
    if (blocking) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }
    if (phEvent) {
      *phEvent = EventPtr.release();
    }
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, const void *pMem, size_t size,
    ur_usm_migration_flags_t flags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = flags;

  void *HIPDevicePtr = const_cast<void *>(pMem);
  ur_device_handle_t Device = hQueue->getDevice();

// HIP_POINTER_ATTRIBUTE_RANGE_SIZE is not an attribute in ROCM < 5,
// so we can't perform this check for such cases.
#if HIP_VERSION_MAJOR >= 5
  unsigned int PointerRangeSize = 0;
  UR_CHECK_ERROR(hipPointerGetAttribute(&PointerRangeSize,
                                        HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                        (hipDeviceptr_t)HIPDevicePtr));
  UR_ASSERT(size <= PointerRangeSize, UR_RESULT_ERROR_INVALID_SIZE);
#endif

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_PREFETCH, hQueue, HIPStream));
      UR_CHECK_ERROR(EventPtr->start());
    }

    // Helper to ensure returning a valid event on early exit.
    auto releaseEvent = [&EventPtr, &phEvent]() -> void {
      if (phEvent) {
        UR_CHECK_ERROR(EventPtr->record());
        *phEvent = EventPtr.release();
      }
    };

    // If the device does not support managed memory access, we can't set
    // mem_advise.
    if (!Device->getManagedMemSupport()) {
      releaseEvent();
      setErrorMessage("mem_advise ignored as device does not support "
                      "managed memory access",
                      UR_RESULT_SUCCESS);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }

    hipPointerAttribute_t attribs;
    // TODO: hipPointerGetAttributes will fail if pMem is non-HIP allocated
    // memory, as it is neither registered as host memory, nor into the
    // address space for the current device, meaning the pMem ptr points to a
    // system-allocated memory. This means we may need to check
    // system-alloacted memory and handle the failure more gracefully.
    UR_CHECK_ERROR(hipPointerGetAttributes(&attribs, pMem));
    // async prefetch requires USM pointer (or hip SVM) to work.
    if (!attribs.isManaged) {
      releaseEvent();
      setErrorMessage("Prefetch hint ignored as prefetch only works with USM",
                      UR_RESULT_SUCCESS);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }

    UR_CHECK_ERROR(
        hipMemPrefetchAsync(pMem, size, hQueue->getDevice()->get(), HIPStream));
    releaseEvent();
  } catch (ur_result_t Err) {
    return Err;
  }

  return UR_RESULT_SUCCESS;
}

/// USM: memadvise API to govern behavior of automatic migration mechanisms
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem, size_t size,
                   ur_usm_advice_flags_t advice, ur_event_handle_t *phEvent) {
  UR_ASSERT(pMem && size > 0, UR_RESULT_ERROR_INVALID_VALUE);
  void *HIPDevicePtr = const_cast<void *>(pMem);
  ur_device_handle_t Device = hQueue->getDevice();

#if HIP_VERSION_MAJOR >= 5
  // NOTE: The hipPointerGetAttribute API is marked as beta, meaning, while
  // this is feature complete, it is still open to changes and outstanding
  // issues.
  size_t PointerRangeSize = 0;
  UR_CHECK_ERROR(hipPointerGetAttribute(
      &PointerRangeSize, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
      static_cast<hipDeviceptr_t>(HIPDevicePtr)));
  UR_ASSERT(size <= PointerRangeSize, UR_RESULT_ERROR_INVALID_SIZE);
#endif

  try {
    ScopedDevice Active(Device);
    std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_ADVISE, hQueue, hQueue->getNextTransferStream()));
      EventPtr->start();
    }

    // Helper to ensure returning a valid event on early exit.
    auto releaseEvent = [&EventPtr, &phEvent]() -> void {
      if (phEvent) {
        UR_CHECK_ERROR(EventPtr->record());
        *phEvent = EventPtr.release();
      }
    };

    // If the device does not support managed memory access, we can't set
    // mem_advise.
    if (!Device->getManagedMemSupport()) {
      releaseEvent();
      setErrorMessage("mem_advise ignored as device does not support "
                      "managed memory access",
                      UR_RESULT_SUCCESS);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }

    // Passing MEM_ADVICE_SET/MEM_ADVICE_CLEAR_PREFERRED_LOCATION to
    // hipMemAdvise on a GPU device requires the GPU device to report a
    // non-zero value for hipDeviceAttributeConcurrentManagedAccess.
    // Therefore, ignore the mem advice if concurrent managed memory access is
    // not available.
    if (advice & (UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION |
                  UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION |
                  UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE |
                  UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE |
                  UR_USM_ADVICE_FLAG_DEFAULT)) {
      if (!Device->getConcurrentManagedAccess()) {
        releaseEvent();
        setErrorMessage("mem_advise ignored as device does not support "
                        "concurrent managed access",
                        UR_RESULT_SUCCESS);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }

      // TODO: If pMem points to valid system-allocated pageable memory, we
      // should check that the device also has the
      // hipDeviceAttributePageableMemoryAccess property, so that a valid
      // read-only copy can be created on the device. This also applies for
      // UR_USM_MEM_ADVICE_SET/MEM_ADVICE_CLEAR_READ_MOSTLY.
    }

    // hipMemAdvise only supports managed memory allocated via
    // hipMallocManaged. We can't support this API with any other types of
    // pointer. We should ignore them and result UR_RESULT_SUCCESS but instead
    // we report a warning.
    // FIXME: Fix this up when there's a better warning mechanism.
    if (auto ptrAttribs = getPointerAttributes(pMem);
        !ptrAttribs || !ptrAttribs->isManaged) {
      releaseEvent();
      setErrorMessage("mem_advise is ignored as the pointer argument is not "
                      "a shared USM pointer",
                      UR_RESULT_SUCCESS);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }

    const auto DeviceID = Device->get();
    if (advice & UR_USM_ADVICE_FLAG_DEFAULT) {
      UR_CHECK_ERROR(
          hipMemAdvise(pMem, size, hipMemAdviseUnsetReadMostly, DeviceID));
      UR_CHECK_ERROR(hipMemAdvise(
          pMem, size, hipMemAdviseUnsetPreferredLocation, DeviceID));
      UR_CHECK_ERROR(
          hipMemAdvise(pMem, size, hipMemAdviseUnsetAccessedBy, DeviceID));
#if defined(__HIP_PLATFORM_AMD__)
      UR_CHECK_ERROR(
          hipMemAdvise(pMem, size, hipMemAdviseUnsetCoarseGrain, DeviceID));
#endif
    } else {
      ur_result_t Result =
          setHipMemAdvise(HIPDevicePtr, size, advice, DeviceID);
      assert((Result == UR_RESULT_SUCCESS ||
              Result == UR_RESULT_ERROR_INVALID_ENUMERATION) &&
             "Unexpected return code");
      // UR_RESULT_ERROR_INVALID_ENUMERATION is returned when using a valid
      // but currently unmapped advice arguments as not supported by this
      // platform. Therefore, warn the user instead of throwing and aborting
      // the runtime.
      if (Result == UR_RESULT_ERROR_INVALID_ENUMERATION) {
        releaseEvent();
        setErrorMessage("mem_advise is ignored as the advice argument is not "
                        "supported by this device",
                        UR_RESULT_SUCCESS);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      UR_CHECK_ERROR(Result);
    }

    releaseEvent();
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t, void *, size_t, size_t, const void *, size_t, size_t,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/// 2D Memcpy API
///
/// \param hQueue is the queue to submit to
/// \param blocking is whether this operation should block the host
/// \param pDst is the location the data will be copied
/// \param dstPitch is the total width of the destination memory including
/// padding
/// \param pSrc is the data to be copied
/// \param srcPitch is the total width of the source memory including padding
/// \param width is width in bytes of each row to be copied
/// \param height is height the columns to be copied
/// \param numEventsInWaitList is the number of events to wait on
/// \param phEventWaitList is an array of events to wait on
/// \param phEvent is the event that represents this operation
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, size_t dstPitch,
    const void *pSrc, size_t srcPitch, size_t width, size_t height,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_MEMCPY_2D, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    // There is an issue with hipMemcpy2D* when hipMemcpyDefault is used,
    // which makes the HIP runtime not correctly derive the copy kind
    // (direction) for the copies since ROCm 5.6.0+. See:
    // https://github.com/ROCm/clr/issues/40
    // Fixed by commit
    // https://github.com/ROCm/clr/commit/d3bfb55d7a934355257a72fab538a0a634b43cad
    // included in releases starting from ROCm 6.1.0.
#if HIP_VERSION >= 50600000 && HIP_VERSION < 60100000
    hipPointerAttribute_t srcAttribs{};
    hipPointerAttribute_t dstAttribs{};

    // Determine if pSrc and/or pDst are system allocated pageable host
    // memory.
    bool srcIsSystemAlloc{false};
    bool dstIsSystemAlloc{false};

    hipError_t hipRes{};
    // Error code hipErrorInvalidValue returned from hipPointerGetAttributes
    // for a non-null pointer refers to an OS-allocation, hence we can work
    // with the assumption that this is a pointer to a pageable host memory.
    // Since ROCm version 6.0.0, the enum hipMemoryType can also be marked as
    // hipMemoryTypeUnregistered explicitly to relay that information better.
    // This means we cannot rely on any attribute result, hence we just mark
    // the pointer handle as system allocated pageable host memory.
    // The HIP runtime can handle the registering/unregistering of the memory
    // as long as the right copy-kind (direction) is provided to hipMemcpy2D*.
    hipRes = hipPointerGetAttributes(&srcAttribs, pSrc);
    if (hipRes == hipErrorInvalidValue && pSrc)
      srcIsSystemAlloc = true;
    hipRes = hipPointerGetAttributes(&dstAttribs, (const void *)pDst);
    if (hipRes == hipErrorInvalidValue && pDst)
      dstIsSystemAlloc = true;
#if HIP_VERSION_MAJOR >= 6
    srcIsSystemAlloc |= srcAttribs.type == hipMemoryTypeUnregistered;
    dstIsSystemAlloc |= dstAttribs.type == hipMemoryTypeUnregistered;
#endif

    unsigned int srcMemType{srcAttribs.type};
    unsigned int dstMemType{dstAttribs.type};

    // ROCm 5.7.1 finally started updating the type attribute member to
    // hipMemoryTypeManaged for shared memory allocations(hipMallocManaged).
    // Hence, we use a separate query that verifies the pointer use via flags.
#if HIP_VERSION >= 50700001
    // Determine the source/destination memory type for shared allocations.
    //
    // NOTE: The hipPointerGetAttribute API is marked as [BETA] and fails with
    // exit code -11 when passing a system allocated pointer to it.
    if (!srcIsSystemAlloc && srcAttribs.isManaged) {
      UR_ASSERT(srcAttribs.hostPointer && srcAttribs.devicePointer,
                UR_RESULT_ERROR_INVALID_VALUE);
      UR_CHECK_ERROR(hipPointerGetAttribute(
          &srcMemType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
          reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(pSrc))));
    }
    if (!dstIsSystemAlloc && dstAttribs.isManaged) {
      UR_ASSERT(dstAttribs.hostPointer && dstAttribs.devicePointer,
                UR_RESULT_ERROR_INVALID_VALUE);
      UR_CHECK_ERROR(
          hipPointerGetAttribute(&dstMemType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                 reinterpret_cast<hipDeviceptr_t>(pDst)));
    }
#endif

    const bool srcIsHost{(srcMemType == hipMemoryTypeHost) || srcIsSystemAlloc};
    const bool srcIsDevice{srcMemType == hipMemoryTypeDevice};
    const bool dstIsHost{(dstMemType == hipMemoryTypeHost) || dstIsSystemAlloc};
    const bool dstIsDevice{dstMemType == hipMemoryTypeDevice};

    unsigned int cpyKind{};
    if (srcIsHost && dstIsHost)
      cpyKind = hipMemcpyHostToHost;
    else if (srcIsHost && dstIsDevice)
      cpyKind = hipMemcpyHostToDevice;
    else if (srcIsDevice && dstIsHost)
      cpyKind = hipMemcpyDeviceToHost;
    else if (srcIsDevice && dstIsDevice)
      cpyKind = hipMemcpyDeviceToDevice;
    else
      cpyKind = hipMemcpyDefault;

    UR_CHECK_ERROR(hipMemcpy2DAsync(pDst, dstPitch, pSrc, srcPitch, width,
                                    height, (hipMemcpyKind)cpyKind, HIPStream));
#else
    UR_CHECK_ERROR(hipMemcpy2DAsync(pDst, dstPitch, pSrc, srcPitch, width,
                                    height, hipMemcpyDefault, HIPStream));
#endif

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
    if (blocking) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }
  } catch (ur_result_t Err) {
    return Err;
  }

  return UR_RESULT_SUCCESS;
}

namespace {

enum class GlobalVariableCopy { Read, Write };

ur_result_t deviceGlobalCopyHelper(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blocking, size_t count, size_t offset, void *ptr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, GlobalVariableCopy CopyType) {

  try {
    hipDeviceptr_t DeviceGlobal = nullptr;
    size_t DeviceGlobalSize = 0;
    UR_CHECK_ERROR(hProgram->getGlobalVariablePointer(name, &DeviceGlobal,
                                                      &DeviceGlobalSize));

    if (offset + count > DeviceGlobalSize)
      return UR_RESULT_ERROR_INVALID_VALUE;

    void *pSrc, *pDst;
    if (CopyType == GlobalVariableCopy::Write) {
      pSrc = ptr;
      pDst = reinterpret_cast<uint8_t *>(DeviceGlobal) + offset;
    } else {
      pSrc = reinterpret_cast<uint8_t *>(DeviceGlobal) + offset;
      pDst = ptr;
    }
    return urEnqueueUSMMemcpy(hQueue, blocking, pDst, pSrc, count,
                              numEventsInWaitList, phEventWaitList, phEvent);
  } catch (ur_result_t Err) {
    return Err;
  }
}
} // namespace

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return deviceGlobalCopyHelper(hQueue, hProgram, name, blockingWrite, count,
                                offset, const_cast<void *>(pSrc),
                                numEventsInWaitList, phEventWaitList, phEvent,
                                GlobalVariableCopy::Write);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  return deviceGlobalCopyHelper(
      hQueue, hProgram, name, blockingRead, count, offset, pDst,
      numEventsInWaitList, phEventWaitList, phEvent, GlobalVariableCopy::Read);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t, ur_program_handle_t, const char *, bool, void *, size_t,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t, ur_program_handle_t, const char *, bool, void *, size_t,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

// Helper to compute kernel parameters from workload
// dimensions.
// @param [in]  Device handler to the target Device
// @param [in]  WorkDim workload dimension
// @param [in]  GlobalWorkOffset pointer workload global offsets
// @param [in]  GlobalWorkSize pointer workload global sizes
// @param [in]  LocalWorkOffset pointer workload local offsets
// @param [inout] Kernel handler to the kernel
// @param [inout] HIPFunc handler to the HIP function attached to the kernel
// @param [out] ThreadsPerBlock Number of threads per block we should run
// @param [out] BlocksPerGrid Number of blocks per grid we should run
ur_result_t
setKernelParams(const ur_device_handle_t Device, const uint32_t WorkDim,
                const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
                const size_t *LocalWorkSize, ur_kernel_handle_t &Kernel,
                hipFunction_t &HIPFunc, size_t (&ThreadsPerBlock)[3],
                size_t (&BlocksPerGrid)[3]) {
  size_t MaxWorkGroupSize = 0;
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedDevice Active(Device);
    {
      size_t MaxThreadsPerBlock[3] = {
          static_cast<size_t>(Device->getMaxBlockDimX()),
          static_cast<size_t>(Device->getMaxBlockDimY()),
          static_cast<size_t>(Device->getMaxBlockDimZ())};

      auto &ReqdThreadsPerBlock = Kernel->ReqdThreadsPerBlock;
      MaxWorkGroupSize = Device->getMaxWorkGroupSize();

      if (LocalWorkSize != nullptr) {
        auto isValid = [&](int dim) {
          UR_ASSERT(ReqdThreadsPerBlock[dim] == 0 ||
                        LocalWorkSize[dim] == ReqdThreadsPerBlock[dim],
                    UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
          UR_ASSERT(LocalWorkSize[dim] <= MaxThreadsPerBlock[dim],
                    UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
          // Checks that local work sizes are a divisor of the global work
          // sizes which includes that the local work sizes are neither larger
          // than the global work sizes and not 0.
          UR_ASSERT(LocalWorkSize != 0,
                    UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
          UR_ASSERT((GlobalWorkSize[dim] % LocalWorkSize[dim]) == 0,
                    UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
          ThreadsPerBlock[dim] = LocalWorkSize[dim];
          return UR_RESULT_SUCCESS;
        };

        for (size_t dim = 0; dim < WorkDim; dim++) {
          auto err = isValid(dim);
          if (err != UR_RESULT_SUCCESS)
            return err;
        }
      } else {
        guessLocalWorkSize(Device, ThreadsPerBlock, GlobalWorkSize, WorkDim,
                           MaxThreadsPerBlock);
      }
    }

    UR_ASSERT(MaxWorkGroupSize >=
                  size_t(ThreadsPerBlock[0] * ThreadsPerBlock[1] *
                         ThreadsPerBlock[2]),
              UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);

    for (size_t i = 0; i < WorkDim; i++) {
      BlocksPerGrid[i] =
          (GlobalWorkSize[i] + ThreadsPerBlock[i] - 1) / ThreadsPerBlock[i];
    }

    // Set the implicit global offset parameter if kernel has offset variant
    if (Kernel->getWithOffsetParameter()) {
      std::uint32_t ImplicitOffset[3] = {0, 0, 0};
      if (GlobalWorkOffset) {
        for (size_t i = 0; i < WorkDim; i++) {
          ImplicitOffset[i] = static_cast<std::uint32_t>(GlobalWorkOffset[i]);
          if (GlobalWorkOffset[i] != 0) {
            HIPFunc = Kernel->getWithOffsetParameter();
          }
        }
      }
      Kernel->setImplicitOffsetArg(sizeof(ImplicitOffset), ImplicitOffset);
    }

    // Set local mem max size if env var is present
    static const char *LocalMemSzPtrUR =
        std::getenv("UR_HIP_MAX_LOCAL_MEM_SIZE");
    static const char *LocalMemSzPtrPI =
        std::getenv("SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE");
    static const char *LocalMemSzPtr =
        LocalMemSzPtrUR ? LocalMemSzPtrUR
                        : (LocalMemSzPtrPI ? LocalMemSzPtrPI : nullptr);

    if (LocalMemSzPtr) {
      int DeviceMaxLocalMem = Device->getDeviceMaxLocalMem();
      static const int EnvVal = std::atoi(LocalMemSzPtr);
      if (EnvVal <= 0 || EnvVal > DeviceMaxLocalMem) {
        setErrorMessage(LocalMemSzPtrUR ? "Invalid value specified for "
                                          "UR_HIP_MAX_LOCAL_MEM_SIZE"
                                        : "Invalid value specified for "
                                          "SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE",
                        UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      UR_CHECK_ERROR(hipFuncSetAttribute(
          HIPFunc, hipFuncAttributeMaxDynamicSharedMemorySize, EnvVal));
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

void setCopyRectParams(ur_rect_region_t Region, const void *SrcPtr,
                       const hipMemoryType SrcType, ur_rect_offset_t SrcOffset,
                       size_t SrcRowPitch, size_t SrcSlicePitch, void *DstPtr,
                       const hipMemoryType DstType, ur_rect_offset_t DstOffset,
                       size_t DstRowPitch, size_t DstSlicePitch,
                       hipMemcpy3DParms &Params) {
  // Set all params to 0 first
  std::memset(&Params, 0, sizeof(hipMemcpy3DParms));

  SrcRowPitch = (!SrcRowPitch) ? Region.width + SrcOffset.x : SrcRowPitch;
  SrcSlicePitch = (!SrcSlicePitch)
                      ? ((Region.height + SrcOffset.y) * SrcRowPitch)
                      : SrcSlicePitch;
  DstRowPitch = (!DstRowPitch) ? Region.width + DstOffset.x : DstRowPitch;
  DstSlicePitch = (!DstSlicePitch)
                      ? ((Region.height + DstOffset.y) * DstRowPitch)
                      : DstSlicePitch;

  Params.extent.depth = Region.depth;
  Params.extent.height = Region.height;
  Params.extent.width = Region.width;

  Params.srcPtr.ptr = const_cast<void *>(SrcPtr);
  Params.srcPtr.pitch = SrcRowPitch;
  Params.srcPtr.xsize = SrcRowPitch;
  Params.srcPtr.ysize = SrcSlicePitch / SrcRowPitch;
  Params.srcPos.x = SrcOffset.x;
  Params.srcPos.y = SrcOffset.y;
  Params.srcPos.z = SrcOffset.z;

  Params.dstPtr.ptr = const_cast<void *>(DstPtr);
  Params.dstPtr.pitch = DstRowPitch;
  Params.dstPtr.xsize = DstRowPitch;
  Params.dstPtr.ysize = DstSlicePitch / DstRowPitch;
  Params.dstPos.x = DstOffset.x;
  Params.dstPos.y = DstOffset.y;
  Params.dstPos.z = DstOffset.z;

  Params.kind = (SrcType == hipMemoryTypeDevice
                     ? (DstType == hipMemoryTypeDevice ? hipMemcpyDeviceToDevice
                                                       : hipMemcpyDeviceToHost)
                     : (DstType == hipMemoryTypeDevice ? hipMemcpyHostToDevice
                                                       : hipMemcpyHostToHost));
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueTimestampRecordingExp(
    ur_queue_handle_t hQueue, bool blocking, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
  try {
    ScopedDevice Active(hQueue->getDevice());

    uint32_t StreamToken;
    ur_stream_guard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                     phEventWaitList));

    RetImplEvent =
        std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
            UR_COMMAND_TIMESTAMP_RECORDING_EXP, hQueue, HIPStream));
    UR_CHECK_ERROR(RetImplEvent->start());
    UR_CHECK_ERROR(RetImplEvent->record());

    if (blocking) {
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    *phEvent = RetImplEvent.release();
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}
