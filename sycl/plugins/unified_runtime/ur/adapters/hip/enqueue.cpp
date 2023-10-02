//===--------- enqueue.cpp - HIP Adapter ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"

namespace {

static size_t imageElementByteSize(hipArray_Format ArrayFormat) {
  switch (ArrayFormat) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
  case HIP_AD_FORMAT_SIGNED_INT8:
    return 1;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
  case HIP_AD_FORMAT_SIGNED_INT16:
  case HIP_AD_FORMAT_HALF:
    return 2;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
  case HIP_AD_FORMAT_SIGNED_INT32:
  case HIP_AD_FORMAT_FLOAT:
    return 4;
  default:
    detail::ur::die("Invalid image format.");
  }
  return 0;
}

ur_result_t enqueueEventsWait(hipStream_t Stream, uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList) {
  if (!EventWaitList) {
    return UR_RESULT_SUCCESS;
  }
  try {
    auto Result = forLatestEvents(
        EventWaitList, NumEventsInWaitList,
        [Stream](ur_event_handle_t Event) -> ur_result_t {
          ScopedDevice Active(Event->getDevice());
          if (Event->isCompleted() || Event->getStream() == Stream) {
            return UR_RESULT_SUCCESS;
          } else {
            return UR_CHECK_ERROR(hipStreamWaitEvent(Stream, Event->get(), 0));
          }
        });

    if (Result != UR_RESULT_SUCCESS) {
      return Result;
    }
    return UR_RESULT_SUCCESS;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

void simpleGuessLocalWorkSize(size_t *ThreadsPerBlock,
                              const size_t *GlobalWorkSize,
                              const size_t MaxThreadsPerBlock[3],
                              ur_kernel_handle_t Kernel) {
  assert(ThreadsPerBlock != nullptr);
  assert(GlobalWorkSize != nullptr);
  assert(Kernel != nullptr);

  std::ignore = Kernel;

  ThreadsPerBlock[0] = std::min(MaxThreadsPerBlock[0], GlobalWorkSize[0]);

  // Find a local work group size that is a divisor of the global
  // work group size to produce uniform work groups.
  while (GlobalWorkSize[0] % ThreadsPerBlock[0]) {
    --ThreadsPerBlock[0];
  }
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
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_buffer_ *Buffer = ur_cast<ur_buffer_ *>(hBuffer);
  Buffer->allocateMemObjOnDeviceIfNeeded(hQueue->getDevice());

  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_WRITE, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Result = UR_CHECK_ERROR(
        hipMemcpyHtoDAsync(Buffer->getWithOffset(offset, hQueue->getDevice()),
                           const_cast<void *>(pSrc), size, HIPStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingWrite) {
      Result = UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_buffer_ *Buffer = ur_cast<ur_buffer_ *>(hBuffer);

  ur_result_t Result = UR_RESULT_SUCCESS;

  ur_lock MemoryMigrationLock(hBuffer->MemoryMigrationMutex);

  // Note that this entry point may be called on a specific queue that may not
  // be the last queue to write to the MemBuffer
  auto DeviceToCopyFrom = Buffer->LastEventWritingToMemObj == nullptr
                              ? hQueue->getDevice()
                              : Buffer->LastEventWritingToMemObj->getDevice();

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(DeviceToCopyFrom);
    // Use the default stream if copying from another device
    hipStream_t HIPStream = DeviceToCopyFrom == hQueue->getDevice()
                                ? hQueue->getNextTransferStream()
                                : hipStream_t{0};

    UR_CHECK_ERROR(
        enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList));
    if (Buffer->LastEventWritingToMemObj != nullptr &&
        hQueue->getDevice() != DeviceToCopyFrom) {
      // We may have to wait for an event on another queue if it is the last
      // event writing to mem obj
      UR_CHECK_ERROR(
          enqueueEventsWait(HIPStream, 1, &Buffer->LastEventWritingToMemObj));
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    if (auto SrcPtr = Buffer->getWithOffset(offset, DeviceToCopyFrom)) {
      UR_CHECK_ERROR(hipMemcpyDtoHAsync(pDst, SrcPtr, size, HIPStream));
    }

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingRead) {
      Result = UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t err) {
    Result = err;
  }
  return Result;
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

  std::vector<ur_event_handle_t> DepEvents(
      phEventWaitList, phEventWaitList + numEventsInWaitList);
  std::vector<std::pair<ur_mem_handle_t, ur_lock>> MemMigrationLocks;

  // phEventWaitList only contains events that are handed to UR by the SYCL
  // runtime. However since UR handles memory dependencies within a context
  // we may need to add more events to our dependent events list if the UR
  // context contains multiple devices
  if (hQueue->getContext()->NumDevices > 1) {
    MemMigrationLocks.reserve(hKernel->Args.MemObjArgs.size());
    for (auto &MemArg : hKernel->Args.MemObjArgs) {
      bool PushBack = false;
      if (auto MemDepEvent = MemArg.Mem->LastEventWritingToMemObj;
          MemDepEvent && std::find(DepEvents.begin(), DepEvents.end(),
                                   MemDepEvent) == DepEvents.end()) {
        DepEvents.push_back(MemDepEvent);
        PushBack = true;
      }
      if ((MemArg.AccessFlags &
           (UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_WRITE_ONLY)) ||
          PushBack) {
        if (std::find_if(MemMigrationLocks.begin(), MemMigrationLocks.end(),
                         [MemArg](auto &Lock) {
                           return Lock.first == MemArg.Mem;
                         }) == MemMigrationLocks.end())
          MemMigrationLocks.emplace_back(
              std::pair{MemArg.Mem, ur_lock{MemArg.Mem->MemoryMigrationMutex}});
      }
    }
  }

  // Early exit for zero size range kernel
  if (*pGlobalWorkSize == 0) {
    if (DepEvents.size()) {
      return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                            &DepEvents[0], phEvent);
    }
    return UR_RESULT_SUCCESS;
  }

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  size_t ThreadsPerBlock[3] = {32u, 1u, 1u};
  size_t MaxWorkGroupSize = 0u;
  size_t MaxThreadsPerBlock[3] = {};
  bool ProvidedLocalWorkGroupSize = (pLocalWorkSize != nullptr);

  {
    ur_result_t Result = urDeviceGetInfo(
        hQueue->Device, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
        sizeof(MaxThreadsPerBlock), MaxThreadsPerBlock, nullptr);
    UR_ASSERT(Result == UR_RESULT_SUCCESS, Result);

    Result =
        urDeviceGetInfo(hQueue->Device, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                        sizeof(MaxWorkGroupSize), &MaxWorkGroupSize, nullptr);
    UR_ASSERT(Result == UR_RESULT_SUCCESS, Result);

    // The MaxWorkGroupSize = 1024 for AMD GPU
    // The MaxThreadsPerBlock = {1024, 1024, 1024}

    if (ProvidedLocalWorkGroupSize) {
      auto isValid = [&](int dim) {
        UR_ASSERT(pLocalWorkSize[dim] <= MaxThreadsPerBlock[dim],
                  UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
        // Checks that local work sizes are a divisor of the global work sizes
        // which includes that the local work sizes are neither larger than the
        // global work sizes and not 0.
        UR_ASSERT(pLocalWorkSize != 0, UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
        UR_ASSERT((pGlobalWorkSize[dim] % pLocalWorkSize[dim]) == 0,
                  UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
        ThreadsPerBlock[dim] = pLocalWorkSize[dim];
        return UR_RESULT_SUCCESS;
      };

      for (size_t dim = 0; dim < workDim; dim++) {
        auto err = isValid(dim);
        if (err != UR_RESULT_SUCCESS)
          return err;
      }
    } else {
      simpleGuessLocalWorkSize(ThreadsPerBlock, pGlobalWorkSize,
                               MaxThreadsPerBlock, hKernel);
    }
  }

  UR_ASSERT(MaxWorkGroupSize >= size_t(ThreadsPerBlock[0] * ThreadsPerBlock[1] *
                                       ThreadsPerBlock[2]),
            UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);

  size_t BlocksPerGrid[3] = {1u, 1u, 1u};

  for (size_t i = 0; i < workDim; i++) {
    BlocksPerGrid[i] =
        (pGlobalWorkSize[i] + ThreadsPerBlock[i] - 1) / ThreadsPerBlock[i];
  }

  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ur_device_handle_t Dev = hQueue->getDevice();
    ScopedDevice Active(Dev);
    ur_context_handle_t Ctx = hQueue->getContext();

    uint32_t StreamToken;
    ur_stream_quard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    hipFunction_t HIPFunc = hKernel->get();

    hipDevice_t HIPDev = Dev->get();
    for (const void *P : hKernel->getPtrArgs()) {
      auto [Addr, Size] = Ctx->getUSMMapping(P);
      if (!Addr)
        continue;
      if (hipMemPrefetchAsync(Addr, Size, HIPDev, HIPStream) != hipSuccess)
        return UR_RESULT_ERROR_INVALID_KERNEL_ARGS;
    }

    if (DepEvents.size()) {
      Result =
          enqueueEventsWait(HIPStream, numEventsInWaitList, DepEvents.data());
    }

    // For memory migration across devices in the same context
    if (hQueue->getContext()->NumDevices > 1) {
      for (auto &MemArg : hKernel->Args.MemObjArgs) {
        MemArg.Mem->migrateMemoryToDeviceIfNeeded(hQueue->getDevice());
      }
    }

    // Set the implicit global offset parameter if kernel has offset variant
    if (hKernel->getWithOffsetParameter()) {
      std::uint32_t hip_implicit_offset[3] = {0, 0, 0};
      if (pGlobalWorkOffset) {
        for (size_t i = 0; i < workDim; i++) {
          hip_implicit_offset[i] =
              static_cast<std::uint32_t>(pGlobalWorkOffset[i]);
          if (pGlobalWorkOffset[i] != 0) {
            HIPFunc = hKernel->getWithOffsetParameter();
          }
        }
      }
      hKernel->setImplicitOffsetArg(sizeof(hip_implicit_offset),
                                    hip_implicit_offset);
    }

    auto ArgIndices = hKernel->getArgIndices();

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_KERNEL_LAUNCH, hQueue, HIPStream, StreamToken));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    // Once event has been started we can unlock MemoryMigrationMutex
    if (hQueue->getContext()->NumDevices > 1) {
      for (auto &MemArg : hKernel->Args.MemObjArgs) {
        // Telling the ur_mem_handle_t that it will need to wait on this kernel
        // if it has been written to
        if (phEvent && (MemArg.AccessFlags &
                        (UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_WRITE_ONLY))) {
          MemArg.Mem->setLastEventWritingToMemObj(RetImplEvent.get());
        }
      }
      // We can release the MemoryMigrationMutexes now
      MemMigrationLocks.clear();
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
      int DeviceMaxLocalMem = 0;
      Result = UR_CHECK_ERROR(hipDeviceGetAttribute(
          &DeviceMaxLocalMem, hipDeviceAttributeMaxSharedMemoryPerBlock,
          HIPDev));

      static const int EnvVal = std::atoi(LocalMemSzPtr);
      if (EnvVal <= 0 || EnvVal > DeviceMaxLocalMem) {
        setErrorMessage(LocalMemSzPtrUR ? "Invalid value specified for "
                                          "UR_HIP_MAX_LOCAL_MEM_SIZE"
                                        : "Invalid value specified for "
                                          "SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE",
                        UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      Result = UR_CHECK_ERROR(hipFuncSetAttribute(
          HIPFunc, hipFuncAttributeMaxDynamicSharedMemorySize, EnvVal));
    }

    Result = UR_CHECK_ERROR(hipModuleLaunchKernel(
        HIPFunc, BlocksPerGrid[0], BlocksPerGrid[1], BlocksPerGrid[2],
        ThreadsPerBlock[0], ThreadsPerBlock[1], ThreadsPerBlock[2],
        hKernel->getLocalSize(), HIPStream, ArgIndices.data(), nullptr));

    hKernel->clearLocalSize();

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t err) {
    Result = err;
  }
  return Result;
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

  ur_result_t Result;

  try {
    ScopedDevice Active(hQueue->getDevice());
    uint32_t StreamToken;
    ur_stream_quard Guard;
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
                return UR_CHECK_ERROR(
                    hipStreamWaitEvent(HIPStream, Event->get(), 0));
              }
            });
      }

      Result = UR_CHECK_ERROR(hipEventRecord(hQueue->BarrierEvent, HIPStream));
      for (unsigned int i = 0; i < hQueue->ComputeAppliedBarrier.size(); i++) {
        hQueue->ComputeAppliedBarrier[i] = false;
      }
      for (unsigned int i = 0; i < hQueue->TransferAppliedBarrier.size(); i++) {
        hQueue->TransferAppliedBarrier[i] = false;
      }
    }
    if (Result != UR_RESULT_SUCCESS) {
      return Result;
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

  return UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&Params, HipStream));
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

  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_buffer_ *Buffer = ur_cast<ur_buffer_ *>(hBuffer);
  ur_lock MemoryMigrationLock(hBuffer->MemoryMigrationMutex);

  ur_result_t Result = UR_RESULT_SUCCESS;

  // Note that this entry point may be called on a specific queue that may not
  // be the last queue to write to the MemBuffer
  auto DeviceToCopyFrom = Buffer->LastEventWritingToMemObj == nullptr
                              ? hQueue->getDevice()
                              : Buffer->LastEventWritingToMemObj->getDevice();

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(DeviceToCopyFrom);
    // Use the default stream if copying from another device
    hipStream_t HIPStream = DeviceToCopyFrom == hQueue->getDevice()
                                ? hQueue->getNextTransferStream()
                                : hipStream_t{0};

    UR_CHECK_ERROR(
        enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList));
    if (Buffer->LastEventWritingToMemObj != nullptr &&
        hQueue->getDevice() != DeviceToCopyFrom) {
      // We may have to wait for an event on another queue if it is the last
      // event writing to mem obj
      UR_CHECK_ERROR(
          enqueueEventsWait(HIPStream, 1, &Buffer->LastEventWritingToMemObj));
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    if (auto SrcPtr = Buffer->getPtr(DeviceToCopyFrom)) {
      UR_CHECK_ERROR(commonEnqueueMemBufferCopyRect(
          HIPStream, region, &SrcPtr, hipMemoryTypeDevice, bufferOrigin,
          bufferRowPitch, bufferSlicePitch, pDst, hipMemoryTypeHost, hostOrigin,
          hostRowPitch, hostSlicePitch));
    }

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingRead) {
      Result = UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_buffer_ *Buffer = ur_cast<ur_buffer_ *>(hBuffer);
  Buffer->allocateMemObjOnDeviceIfNeeded(hQueue->getDevice());
  hipDeviceptr_t DevPtr = Buffer->getPtr(hQueue->getDevice());

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_WRITE_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Result = commonEnqueueMemBufferCopyRect(
        HIPStream, region, pSrc, hipMemoryTypeHost, hostOrigin, hostRowPitch,
        hostSlicePitch, &DevPtr, hipMemoryTypeDevice, bufferOrigin,
        bufferRowPitch, bufferSlicePitch);

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingWrite) {
      Result = UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hBufferSrc->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hBufferDst->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_buffer_ *BufferSrc = ur_cast<ur_buffer_ *>(hBufferSrc);
  ur_buffer_ *BufferDst = ur_cast<ur_buffer_ *>(hBufferDst);

  UR_ASSERT(size + srcOffset <= BufferSrc->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(size + dstOffset <= BufferDst->getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    ur_result_t Result;
    auto Stream = hQueue->getNextTransferStream();

    if (phEventWaitList) {
      Result = enqueueEventsWait(Stream, numEventsInWaitList, phEventWaitList);
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY, hQueue, Stream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    BufferDst->allocateMemObjOnDeviceIfNeeded(hQueue->getDevice());

    auto Src = BufferSrc->getWithOffset(srcOffset, hQueue->getDevice());
    auto Dst = BufferDst->getWithOffset(dstOffset, hQueue->getDevice());

    Result = UR_CHECK_ERROR(hipMemcpyDtoDAsync(Dst, Src, size, Stream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

    return Result;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopyRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  UR_ASSERT(hBufferSrc->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hBufferDst->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_buffer_ *BufferSrc = ur_cast<ur_buffer_ *>(hBufferSrc);
  ur_buffer_ *BufferDst = ur_cast<ur_buffer_ *>(hBufferDst);
  BufferDst->allocateMemObjOnDeviceIfNeeded(hQueue->getDevice());

  void *SrcPtr = BufferSrc->getPtr(hQueue->getDevice());
  void *DstPtr = BufferDst->getPtr(hQueue->getDevice());
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Result = commonEnqueueMemBufferCopyRect(
        HIPStream, region, &SrcPtr, hipMemoryTypeDevice, srcOrigin, srcRowPitch,
        srcSlicePitch, &DstPtr, hipMemoryTypeDevice, dstOrigin, dstRowPitch,
        dstSlicePitch);

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

// HIP has no memset functions that allow setting values more than 4 bytes. UR
// API lets you pass an arbitrary "pattern" to the buffer fill, which can be
// more than 4 bytes. We must break up the pattern into 1 byte values, and set
// the buffer using multiple strided calls.  The first 4 patterns are set using
// hipMemsetD32Async then all subsequent 1 byte patterns are set using
// hipMemset2DAsync which is called for each pattern.
ur_result_t commonMemSetLargePattern(hipStream_t Stream, uint32_t PatternSize,
                                     size_t Size, const void *pPattern,
                                     hipDeviceptr_t Ptr) {
  // Calculate the number of patterns, stride, number of times the pattern
  // needs to be applied, and the number of times the first 32 bit pattern
  // needs to be applied.
  auto NumberOfSteps = PatternSize / sizeof(uint8_t);
  auto Pitch = NumberOfSteps * sizeof(uint8_t);
  auto Height = Size / NumberOfSteps;
  auto Count32 = Size / sizeof(uint32_t);

  // Get 4-byte chunk of the pattern and call hipMemsetD32Async
  auto Value = *(static_cast<const uint32_t *>(pPattern));
  auto Result = UR_CHECK_ERROR(hipMemsetD32Async(Ptr, Value, Count32, Stream));
  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }
  for (auto step = 4u; step < NumberOfSteps; ++step) {
    // take 1 byte of the pattern
    Value = *(static_cast<const uint8_t *>(pPattern) + step);

    // offset the pointer to the part of the buffer we want to write to
    auto OffsetPtr = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(Ptr) +
                                              (step * sizeof(uint8_t)));

    // set all of the pattern chunks
    Result = UR_CHECK_ERROR(hipMemset2DAsync(OffsetPtr, Pitch, Value,
                                             sizeof(uint8_t), Height, Stream));
    if (Result != UR_RESULT_SUCCESS) {
      return Result;
    }
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, const void *pPattern,
    size_t patternSize, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_buffer_ *Buffer = ur_cast<ur_buffer_ *>(hBuffer);
  UR_ASSERT(size + offset <= Buffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);
  auto ArgsAreMultiplesOfPatternSize =
      (offset % patternSize == 0) || (size % patternSize == 0);

  auto PatternIsValid = (pPattern != nullptr);

  auto PatternSizeIsValid =
      ((patternSize & (patternSize - 1)) == 0) && // is power of two
      (patternSize > 0) && (patternSize <= 128);  // falls within valid range

  UR_ASSERT(ArgsAreMultiplesOfPatternSize && PatternIsValid &&
                PatternSizeIsValid,
            UR_RESULT_ERROR_INVALID_VALUE);
  std::ignore = ArgsAreMultiplesOfPatternSize;
  std::ignore = PatternIsValid;
  std::ignore = PatternSizeIsValid;

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());

    auto Stream = hQueue->getNextTransferStream();
    ur_result_t Result;
    if (phEventWaitList) {
      Result = enqueueEventsWait(Stream, numEventsInWaitList, phEventWaitList);
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_FILL, hQueue, Stream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Buffer->allocateMemObjOnDeviceIfNeeded(hQueue->getDevice());
    auto DstDevice = Buffer->getWithOffset(offset, hQueue->getDevice());
    auto N = size / patternSize;

    // pattern size in bytes
    switch (patternSize) {
    case 1: {
      auto Value = *static_cast<const uint8_t *>(pPattern);
      Result = UR_CHECK_ERROR(hipMemsetD8Async(DstDevice, Value, N, Stream));
      break;
    }
    case 2: {
      auto Value = *static_cast<const uint16_t *>(pPattern);
      Result = UR_CHECK_ERROR(hipMemsetD16Async(DstDevice, Value, N, Stream));
      break;
    }
    case 4: {
      auto Value = *static_cast<const uint32_t *>(pPattern);
      Result = UR_CHECK_ERROR(hipMemsetD32Async(DstDevice, Value, N, Stream));
      break;
    }

    default: {
      Result = commonMemSetLargePattern(Stream, patternSize, size, pPattern,
                                        DstDevice);
      break;
    }
    }

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

    return Result;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
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

  if (ImgType == UR_MEM_TYPE_IMAGE2D) {
    hip_Memcpy2D CpyDesc;
    memset(&CpyDesc, 0, sizeof(CpyDesc));
    CpyDesc.srcMemoryType = SrcType;
    if (SrcType == hipMemoryTypeArray) {
      CpyDesc.srcArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(SrcPtr));
      CpyDesc.srcXInBytes = SrcOffset[0];
      CpyDesc.srcY = SrcOffset[1];
    } else {
      CpyDesc.srcHost = SrcPtr;
    }
    CpyDesc.dstMemoryType = DstType;
    if (DstType == hipMemoryTypeArray) {
      CpyDesc.dstArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(DstPtr));
      CpyDesc.dstXInBytes = DstOffset[0];
      CpyDesc.dstY = DstOffset[1];
    } else {
      CpyDesc.dstHost = DstPtr;
    }
    CpyDesc.WidthInBytes = Region[0];
    CpyDesc.Height = Region[1];
    return UR_CHECK_ERROR(hipMemcpyParam2DAsync(&CpyDesc, HipStream));
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
    return UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&CpyDesc, HipStream));
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_ERROR_INVALID_VALUE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingRead,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t, size_t,
    void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(hImage->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_image_ *Image = ur_cast<ur_image_ *>(hImage);

  ur_result_t Result = UR_RESULT_SUCCESS;

  ur_lock MemoryMigrationLock(hImage->MemoryMigrationMutex);

  // Note that this entry point may be called on a specific queue that may not
  // be the last queue to write to the MemBuffer
  auto DeviceToCopyFrom = Image->LastEventWritingToMemObj == nullptr
                              ? hQueue->getDevice()
                              : Image->LastEventWritingToMemObj->getDevice();

  try {
    ScopedDevice Active(DeviceToCopyFrom);
    // Use the default stream if copying from another device
    hipStream_t HIPStream = DeviceToCopyFrom == hQueue->getDevice()
                                ? hQueue->getNextTransferStream()
                                : hipStream_t{0};

    if (phEventWaitList) {
      UR_CHECK_ERROR(
          enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList));
    }

    hipArray *Array = Image->getArray(DeviceToCopyFrom);

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(Array, Format, NumChannels);

    int ElementByteSize = imageElementByteSize(Format);

    size_t ByteOffsetX = origin.x * ElementByteSize * NumChannels;
    size_t BytesToCopy = ElementByteSize * NumChannels * region.depth;

    auto ImgType = Image->getImageType();

    size_t AdjustedRegion[3] = {BytesToCopy, region.height, region.height};
    size_t SrcOffset[3] = {ByteOffsetX, origin.y, origin.z};

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    if (Array != nullptr) {
      UR_CHECK_ERROR(commonEnqueueMemImageNDCopy(
          HIPStream, ImgType, AdjustedRegion, Array, hipMemoryTypeArray,
          SrcOffset, pDst, hipMemoryTypeHost, nullptr));
    }

    if (Result != UR_RESULT_SUCCESS) {
      return Result;
    }

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

    if (blockingRead) {
      Result = UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t, size_t,
    void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(hImage->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  ur_image_ *Image = ur_cast<ur_image_ *>(hImage);
  Image->allocateMemObjOnDeviceIfNeeded(hQueue->getDevice());

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();

    if (phEventWaitList) {
      UR_CHECK_ERROR(
          enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList));
    }

    hipArray *Array = Image->getArray(hQueue->getDevice());

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(Array, Format, NumChannels);

    int ElementByteSize = imageElementByteSize(Format);

    size_t ByteOffsetX = origin.x * ElementByteSize * NumChannels;
    size_t BytesToCopy = ElementByteSize * NumChannels * region.depth;

    auto ImgType = Image->getImageType();

    size_t AdjustedRegion[3] = {BytesToCopy, region.height, region.height};
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
  ur_image_ *ImageSrc = ur_cast<ur_image_ *>(hImageSrc);
  ur_image_ *ImageDst = ur_cast<ur_image_ *>(hImageDst);
  UR_ASSERT(ImageSrc->getImageType() == ImageDst->getImageType(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ImageDst->allocateMemObjOnDeviceIfNeeded(hQueue->getDevice());

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    if (phEventWaitList) {
      UR_CHECK_ERROR(
          enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList));
    }

    hipArray *SrcArray = ImageSrc->getArray(hQueue->getDevice());
    hipArray_Format SrcFormat;
    size_t SrcNumChannels;
    getArrayDesc(SrcArray, SrcFormat, SrcNumChannels);

    hipArray *DstArray = ImageDst->getArray(hQueue->getDevice());
    hipArray_Format DstFormat;
    size_t DstNumChannels;
    getArrayDesc(DstArray, DstFormat, DstNumChannels);

    UR_ASSERT(SrcFormat == DstFormat,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    UR_ASSERT(SrcNumChannels == DstNumChannels,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

    int ElementByteSize = imageElementByteSize(SrcFormat);

    size_t DstByteOffsetX = dstOrigin.x * ElementByteSize * SrcNumChannels;
    size_t SrcByteOffsetX = srcOrigin.x * ElementByteSize * DstNumChannels;
    size_t BytesToCopy = ElementByteSize * SrcNumChannels * region.depth;

    auto ImgType = ImageSrc->getImageType();

    size_t AdjustedRegion[3] = {BytesToCopy, region.height, region.width};
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
  ur_buffer_ *Buffer = ur_cast<ur_buffer_ *>(hBuffer);
  UR_ASSERT(offset + size <= Buffer->getSize(), UR_RESULT_ERROR_INVALID_SIZE);

  ur_result_t Result = UR_RESULT_ERROR_INVALID_OPERATION;
  const bool IsPinned =
      Buffer->MemAllocMode == ur_buffer_::AllocMode::AllocHostPtr;

  // Currently no support for overlapping regions
  if (Buffer->getMapPtr() != nullptr) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // Allocate a pointer in the host to store the mapped information
  auto HostPtr = Buffer->mapToPtr(size, offset, mapFlags);
  *ppRetMap = Buffer->getMapPtr();
  if (HostPtr) {
    Result = UR_RESULT_SUCCESS;
  }

  if (!IsPinned &&
      ((mapFlags & UR_MAP_FLAG_READ) || (mapFlags & UR_MAP_FLAG_WRITE))) {
    // Pinned host memory is already on host so it doesn't need to be read.
    UR_CHECK_ERROR(urEnqueueMemBufferRead(hQueue, hBuffer, blockingMap, offset,
                                          size, HostPtr, numEventsInWaitList,
                                          phEventWaitList, phEvent));
  } else {
    ScopedDevice Active(hQueue->getDevice());

    if (IsPinned) {
      UR_CHECK_ERROR(urEnqueueEventsWait(hQueue, numEventsInWaitList,
                                         phEventWaitList, nullptr));
    }

    if (phEvent) {
      try {
        *phEvent = ur_event_handle_t_::makeNative(
            UR_COMMAND_MEM_BUFFER_MAP, hQueue, hQueue->getNextTransferStream());
        UR_CHECK_ERROR((*phEvent)->start());
        UR_CHECK_ERROR((*phEvent)->record());
      } catch (ur_result_t Error) {
        Result = Error;
      }
    }
  }

  return Result;
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
  ur_buffer_ *Mem = ur_cast<ur_buffer_ *>(hMem);
  UR_ASSERT(Mem->getMapPtr() != nullptr, UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(Mem->getMapPtr() == pMappedPtr, UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  const bool IsPinned =
      Mem->MemAllocMode == ur_buffer_::AllocMode::AllocHostPtr;

  if (!IsPinned &&
      ((Mem->getMapFlags() & UR_MAP_FLAG_WRITE) ||
       (Mem->getMapFlags() & UR_MAP_FLAG_WRITE_INVALIDATE_REGION))) {
    // Pinned host memory is only on host so it doesn't need to be written to.
    UR_CHECK_ERROR(urEnqueueMemBufferWrite(
        hQueue, hMem, true, Mem->getMapOffset(), Mem->getMapSize(), pMappedPtr,
        numEventsInWaitList, phEventWaitList, phEvent));
  } else {
    ScopedDevice Active(hQueue->getDevice());

    if (IsPinned) {
      UR_CHECK_ERROR(urEnqueueEventsWait(hQueue, numEventsInWaitList,
                                         phEventWaitList, nullptr));
    }

    if (phEvent) {
      try {
        *phEvent = ur_event_handle_t_::makeNative(
            UR_COMMAND_MEM_UNMAP, hQueue, hQueue->getNextTransferStream());
        UR_CHECK_ERROR((*phEvent)->start());
        UR_CHECK_ERROR((*phEvent)->record());
      } catch (ur_result_t Error) {
        return Error;
      }
    }
  }

  Mem->unmap(pMappedPtr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    uint32_t StreamToken;
    ur_stream_quard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    Result = enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList);
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_FILL, hQueue, HIPStream, StreamToken));
      UR_CHECK_ERROR(EventPtr->start());
    }

    auto N = size / patternSize;
    switch (patternSize) {
    case 1:
      Result = UR_CHECK_ERROR(
          hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(ptr),
                           *(const uint8_t *)pPattern & 0xFF, N, HIPStream));
      break;
    case 2:
      Result = UR_CHECK_ERROR(hipMemsetD16Async(
          reinterpret_cast<hipDeviceptr_t>(ptr),
          *(const uint16_t *)pPattern & 0xFFFF, N, HIPStream));
      break;
    case 4:
      Result = UR_CHECK_ERROR(hipMemsetD32Async(
          reinterpret_cast<hipDeviceptr_t>(ptr),
          *(const uint32_t *)pPattern & 0xFFFFFFFF, N, HIPStream));
      break;

    default:
      Result = commonMemSetLargePattern(HIPStream, patternSize, size, pPattern,
                                        reinterpret_cast<hipDeviceptr_t>(ptr));
      break;
    }

    if (phEvent) {
      Result = UR_CHECK_ERROR(EventPtr->record());
      *phEvent = EventPtr.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, const void *pSrc,
    size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList);
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_MEMCPY, hQueue, HIPStream));
      UR_CHECK_ERROR(EventPtr->start());
    }
    Result = UR_CHECK_ERROR(
        hipMemcpyAsync(pDst, pSrc, size, hipMemcpyDefault, HIPStream));
    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
    }
    if (blocking) {
      Result = UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }
    if (phEvent) {
      *phEvent = EventPtr.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, const void *pMem, size_t size,
    ur_usm_migration_flags_t flags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
#if HIP_VERSION_MAJOR >= 5
  void *HIPDevicePtr = const_cast<void *>(pMem);
  ur_device_handle_t Device = hQueue->getDevice();

  // If the device does not support managed memory access, we can't set
  // mem_advise.
  if (!getAttribute(Device, hipDeviceAttributeManagedMemory)) {
    setErrorMessage("mem_advise ignored as device does not support "
                    " managed memory access",
                    UR_RESULT_SUCCESS);
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  hipPointerAttribute_t attribs;
  // TODO: hipPointerGetAttributes will fail if pMem is non-HIP allocated
  // memory, as it is neither registered as host memory, nor into the address
  // space for the current device, meaning the pMem ptr points to a
  // system-allocated memory. This means we may need to check system-alloacted
  // memory and handle the failure more gracefully.
  UR_CHECK_ERROR(hipPointerGetAttributes(&attribs, pMem));
  // async prefetch requires USM pointer (or hip SVM) to work.
  if (!attribs.isManaged) {
    setErrorMessage("Prefetch hint ignored as prefetch only works with USM",
                    UR_RESULT_SUCCESS);
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  unsigned int PointerRangeSize = 0;
  UR_CHECK_ERROR(hipPointerGetAttribute(&PointerRangeSize,
                                        HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                        (hipDeviceptr_t)HIPDevicePtr));
  UR_ASSERT(size <= PointerRangeSize, UR_RESULT_ERROR_INVALID_SIZE);

  // flags is currently unused so fail if set
  if (flags != 0)
    return UR_RESULT_ERROR_INVALID_VALUE;
  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList);
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_PREFETCH, hQueue, HIPStream));
      UR_CHECK_ERROR(EventPtr->start());
    }
    Result = UR_CHECK_ERROR(
        hipMemPrefetchAsync(pMem, size, hQueue->getDevice()->get(), HIPStream));
    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
      *phEvent = EventPtr.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }

  return Result;
#else
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif
}

UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem, size_t size,
                   ur_usm_advice_flags_t, ur_event_handle_t *phEvent) {
#if HIP_VERSION_MAJOR >= 5
  void *HIPDevicePtr = const_cast<void *>(pMem);
  unsigned int PointerRangeSize = 0;
  UR_CHECK_ERROR(hipPointerGetAttribute(&PointerRangeSize,
                                        HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                        (hipDeviceptr_t)HIPDevicePtr));
  UR_ASSERT(size <= PointerRangeSize, UR_RESULT_ERROR_INVALID_SIZE);

  // TODO implement a mapping to hipMemAdvise once the expected behaviour
  // of urEnqueueUSMAdvise is detailed in the USM extension
  return urEnqueueEventsWait(hQueue, 0, nullptr, phEvent);
#else
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif
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
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(HIPStream, numEventsInWaitList, phEventWaitList);

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_MEMCPY_2D, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Result =
        UR_CHECK_ERROR(hipMemcpy2DAsync(pDst, dstPitch, pSrc, srcPitch, width,
                                        height, hipMemcpyDefault, HIPStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
    if (blocking) {
      Result = UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t, ur_program_handle_t, const char *, bool, size_t, size_t,
    const void *, uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t, ur_program_handle_t, const char *, bool, size_t, size_t,
    void *, uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
