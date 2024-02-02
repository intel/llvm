//===--------- enqueue.cpp - HIP Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"

extern size_t imageElementByteSize(hipArray_Format ArrayFormat);

namespace {

ur_result_t enqueueEventsWait(ur_queue_handle_t, hipStream_t Stream,
                              uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList) {
  if (!EventWaitList) {
    return UR_RESULT_SUCCESS;
  }
  try {
    auto Result = forLatestEvents(
        EventWaitList, NumEventsInWaitList,
        [Stream](ur_event_handle_t Event) -> ur_result_t {
          ScopedContext Active(Event->getDevice());
          if (Event->isCompleted() || Event->getStream() == Stream) {
            return UR_RESULT_SUCCESS;
          } else {
            UR_CHECK_ERROR(hipStreamWaitEvent(Stream, Event->get(), 0));
            return UR_RESULT_SUCCESS;
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

ur_result_t setHipMemAdvise(const void *DevPtr, const size_t Size,
                            ur_usm_advice_flags_t URAdviceFlags,
                            hipDevice_t Device) {
  // Handle unmapped memory advice flags
  if (URAdviceFlags &
      (UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY |
       UR_USM_ADVICE_FLAG_CLEAR_NON_ATOMIC_MOSTLY |
       UR_USM_ADVICE_FLAG_BIAS_CACHED | UR_USM_ADVICE_FLAG_BIAS_UNCACHED)) {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  using ur_to_hip_advice_t = std::pair<ur_usm_advice_flags_t, hipMemoryAdvise>;

  static constexpr std::array<ur_to_hip_advice_t, 6>
      URToHIPMemAdviseDeviceFlags{
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
      };
  for (auto &FlagPair : URToHIPMemAdviseDeviceFlags) {
    if (URAdviceFlags & FlagPair.first) {
      UR_CHECK_ERROR(hipMemAdvise(DevPtr, Size, FlagPair.second, Device));
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

  for (auto &FlagPair : URToHIPMemAdviseHostFlags) {
    if (URAdviceFlags & FlagPair.first) {
      UR_CHECK_ERROR(
          hipMemAdvise(DevPtr, Size, FlagPair.second, hipCpuDeviceId));
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

  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getDevice());
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
  UR_ASSERT(hBuffer->isBuffer(), UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  ur_lock MemoryMigrationLock{hBuffer->MemoryMigrationMutex};
  auto Device = hQueue->getDevice();
  hipStream_t HIPStream = hQueue->getNextTransferStream();

  try {
    // Note that this entry point may be called on a queue that may not be the
    // last queue to write to the MemBuffer, meaning we must perform the copy
    // from a different device
    if (hBuffer->LastEventWritingToMemObj &&
        hBuffer->LastEventWritingToMemObj->getDevice() != hQueue->getDevice()) {
      Device = hBuffer->LastEventWritingToMemObj->getDevice();
      ScopedContext Active(Device);
      HIPStream = hipStream_t{0}; // Default stream for different device
      // We may have to wait for an event on another queue if it is the last
      // event writing to mem obj
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, 1,
                                       &hBuffer->LastEventWritingToMemObj));
    }

    ScopedContext Active(Device);

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

  std::vector<ur_event_handle_t> DepEvents(
      phEventWaitList, phEventWaitList + numEventsInWaitList);
  std::vector<std::pair<ur_mem_handle_t, ur_lock>> MemMigrationLocks;

  // phEventWaitList only contains events that are handed to UR by the SYCL
  // runtime. However since UR handles memory dependencies within a context
  // we may need to add more events to our dependent events list if the UR
  // context contains multiple devices
  if (hQueue->getContext()->Devices.size() > 1) {
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
      return urEnqueueEventsWaitWithBarrier(hQueue, DepEvents.size(),
                                            phEventWaitList, phEvent);
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
    MaxThreadsPerBlock[0] = hQueue->Device->getMaxBlockDimX();
    MaxThreadsPerBlock[1] = hQueue->Device->getMaxBlockDimY();
    MaxThreadsPerBlock[2] = hQueue->Device->getMaxBlockDimZ();

    MaxWorkGroupSize = hQueue->Device->getMaxWorkGroupSize();

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
    ScopedContext Active(Dev);

    uint32_t StreamToken;
    ur_stream_quard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    hipFunction_t HIPFunc = hKernel->get();

    if (DepEvents.size()) {
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, DepEvents.size(),
                                       DepEvents.data()));
    }

    // For memory migration across devices in the same context
    if (hQueue->getContext()->Devices.size() > 1) {
      for (auto &MemArg : hKernel->Args.MemObjArgs) {
        migrateMemoryToDeviceIfNeeded(MemArg.Mem, hQueue->getDevice());
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
    if (hQueue->getContext()->Devices.size() > 1) {
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
      int DeviceMaxLocalMem = Dev->getDeviceMaxLocalMem();
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

  try {
    ScopedContext Active(hQueue->getDevice());
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

  ur_result_t Result = UR_RESULT_SUCCESS;
  ur_lock MemoryMigrationLock(hBuffer->MemoryMigrationMutex);
  auto Device = hQueue->getDevice();
  hipStream_t HIPStream = hQueue->getNextTransferStream();

  try {
    // Note that this entry point may be called on a queue that may not be the
    // last queue to write to the MemBuffer, meaning we must perform the copy
    // from a different device
    if (hBuffer->LastEventWritingToMemObj &&
        hBuffer->LastEventWritingToMemObj->getDevice() != hQueue->getDevice()) {
      Device = hBuffer->LastEventWritingToMemObj->getDevice();
      ScopedContext Active(Device);
      HIPStream = hipStream_t{0}; // Default stream for different device
      // We may have to wait for an event on another queue if it is the last
      // event writing to mem obj
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, 1,
                                       &hBuffer->LastEventWritingToMemObj));
    }

    ScopedContext Active(Device);

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
  void *DevPtr = std::get<BufferMem>(hBuffer->Mem).getVoid(hQueue->getDevice());
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                               phEventWaitList);

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
      UR_CHECK_ERROR(hipStreamSynchronize(HIPStream));
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
  UR_ASSERT(size + srcOffset <= std::get<BufferMem>(hBufferSrc->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(size + dstOffset <= std::get<BufferMem>(hBufferDst->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getDevice());
    ur_result_t Result = UR_RESULT_SUCCESS;
    auto Stream = hQueue->getNextTransferStream();

    if (phEventWaitList) {
      Result = enqueueEventsWait(hQueue, Stream, numEventsInWaitList,
                                 phEventWaitList);
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
  void *SrcPtr =
      std::get<BufferMem>(hBufferSrc->Mem).getVoid(hQueue->getDevice());
  void *DstPtr =
      std::get<BufferMem>(hBufferDst->Mem).getVoid(hQueue->getDevice());
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                               phEventWaitList);

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
  UR_CHECK_ERROR(hipMemsetD32Async(Ptr, Value, Count32, Stream));
  for (auto step = 4u; step < NumberOfSteps; ++step) {
    // take 1 byte of the pattern
    Value = *(static_cast<const uint8_t *>(pPattern) + step);

    // offset the pointer to the part of the buffer we want to write to
    auto OffsetPtr = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(Ptr) +
                                              (step * sizeof(uint8_t)));

    // set all of the pattern chunks
    UR_CHECK_ERROR(hipMemset2DAsync(OffsetPtr, Pitch, Value, sizeof(uint8_t),
                                    Height, Stream));
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, const void *pPattern,
    size_t patternSize, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(size + offset <= std::get<BufferMem>(hBuffer->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
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
    ScopedContext Active(hQueue->getDevice());

    auto Stream = hQueue->getNextTransferStream();
    ur_result_t Result = UR_RESULT_SUCCESS;
    if (phEventWaitList) {
      Result = enqueueEventsWait(hQueue, Stream, numEventsInWaitList,
                                 phEventWaitList);
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_FILL, hQueue, Stream));
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

  ur_lock MemoryMigrationLock{hImage->MemoryMigrationMutex};
  auto Device = hQueue->getDevice();
  hipStream_t HIPStream = hQueue->getNextTransferStream();

  try {
    // Note that this entry point may be called on a queue that may not be the
    // last queue to write to the MemBuffer, meaning we must perform the copy
    // from a different device
    if (hImage->LastEventWritingToMemObj &&
        hImage->LastEventWritingToMemObj->getDevice() != hQueue->getDevice()) {
      Device = hImage->LastEventWritingToMemObj->getDevice();
      ScopedContext Active(Device);
      HIPStream = hipStream_t{0}; // Default stream for different device
      // We may have to wait for an event on another queue if it is the last
      // event writing to mem obj
      UR_CHECK_ERROR(enqueueEventsWait(hQueue, HIPStream, 1,
                                       &hImage->LastEventWritingToMemObj));
    }

    ScopedContext Active(Device);

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
    size_t BytesToCopy = ElementByteSize * NumChannels * region.depth;

    auto ImgType = std::get<SurfaceMem>(hImage->Mem).getImageType();

    size_t AdjustedRegion[3] = {BytesToCopy, region.height, region.height};
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
    ScopedContext Active(hQueue->getDevice());
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
    size_t BytesToCopy = ElementByteSize * NumChannels * region.depth;

    auto ImgType = std::get<SurfaceMem>(hImage->Mem).getImageType();

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
  UR_ASSERT(std::get<SurfaceMem>(hImageSrc->Mem).getImageType() ==
                std::get<SurfaceMem>(hImageDst->Mem).getImageType(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    if (phEventWaitList) {
      Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                 phEventWaitList);
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

    size_t DstByteOffsetX = dstOrigin.x * ElementByteSize * SrcNumChannels;
    size_t SrcByteOffsetX = srcOrigin.x * ElementByteSize * DstNumChannels;
    size_t BytesToCopy = ElementByteSize * SrcNumChannels * region.depth;

    auto ImgType = std::get<SurfaceMem>(hImageSrc->Mem).getImageType();

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

    Result = commonEnqueueMemImageNDCopy(
        HIPStream, ImgType, AdjustedRegion, SrcArray, hipMemoryTypeArray,
        SrcOffset, DstArray, hipMemoryTypeArray, DstOffset);

    if (Result != UR_RESULT_SUCCESS) {
      return Result;
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

  ur_result_t Result = UR_RESULT_ERROR_INVALID_OPERATION;
  const bool IsPinned =
      BufferImpl.MemAllocMode == BufferMem::AllocMode::AllocHostPtr;

  // Currently no support for overlapping regions
  if (BufferImpl.getMapPtr() != nullptr) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // Allocate a pointer in the host to store the mapped information
  auto HostPtr = BufferImpl.mapToPtr(size, offset, mapFlags);
  *ppRetMap = std::get<BufferMem>(hBuffer->Mem).getMapPtr();
  if (HostPtr) {
    Result = UR_RESULT_SUCCESS;
  }

  if (!IsPinned &&
      ((mapFlags & UR_MAP_FLAG_READ) || (mapFlags & UR_MAP_FLAG_WRITE))) {
    // Pinned host memory is already on host so it doesn't need to be read.
    Result = urEnqueueMemBufferRead(hQueue, hBuffer, blockingMap, offset, size,
                                    HostPtr, numEventsInWaitList,
                                    phEventWaitList, phEvent);
  } else {
    ScopedContext Active(hQueue->getDevice());

    if (IsPinned) {
      Result = urEnqueueEventsWait(hQueue, numEventsInWaitList, phEventWaitList,
                                   nullptr);
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
  ur_result_t Result = UR_RESULT_SUCCESS;
  UR_ASSERT(hMem->isBuffer(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(std::get<BufferMem>(hMem->Mem).getMapPtr() != nullptr,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(std::get<BufferMem>(hMem->Mem).getMapPtr() == pMappedPtr,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  const bool IsPinned = std::get<BufferMem>(hMem->Mem).MemAllocMode ==
                        BufferMem::AllocMode::AllocHostPtr;

  if (!IsPinned &&
      ((std::get<BufferMem>(hMem->Mem).getMapFlags() & UR_MAP_FLAG_WRITE) ||
       (std::get<BufferMem>(hMem->Mem).getMapFlags() &
        UR_MAP_FLAG_WRITE_INVALIDATE_REGION))) {
    // Pinned host memory is only on host so it doesn't need to be written to.
    Result = urEnqueueMemBufferWrite(
        hQueue, hMem, true, std::get<BufferMem>(hMem->Mem).getMapOffset(),
        std::get<BufferMem>(hMem->Mem).getMapSize(), pMappedPtr,
        numEventsInWaitList, phEventWaitList, phEvent);
  } else {
    ScopedContext Active(hQueue->getDevice());

    if (IsPinned) {
      Result = urEnqueueEventsWait(hQueue, numEventsInWaitList, phEventWaitList,
                                   nullptr);
    }

    if (phEvent) {
      try {
        *phEvent = ur_event_handle_t_::makeNative(
            UR_COMMAND_MEM_UNMAP, hQueue, hQueue->getNextTransferStream());
        UR_CHECK_ERROR((*phEvent)->start());
        UR_CHECK_ERROR((*phEvent)->record());
      } catch (ur_result_t Error) {
        Result = Error;
      }
    }
  }

  std::get<BufferMem>(hMem->Mem).unmap(pMappedPtr);
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedContext Active(hQueue->getDevice());
    uint32_t StreamToken;
    ur_stream_quard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                               phEventWaitList);
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
      Result = commonMemSetLargePattern(HIPStream, patternSize, size, pPattern,
                                        reinterpret_cast<hipDeviceptr_t>(ptr));
      break;
    }

    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
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
    ScopedContext Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                               phEventWaitList);
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
    Result = Err;
  }
  return Result;
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

  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                               phEventWaitList);

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
    // memory, as it is neither registered as host memory, nor into the address
    // space for the current device, meaning the pMem ptr points to a
    // system-allocated memory. This means we may need to check system-alloacted
    // memory and handle the failure more gracefully.
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
    Result = Err;
  }

  return Result;
}

/// USM: memadvise API to govern behavior of automatic migration mechanisms
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem, size_t size,
                   ur_usm_advice_flags_t advice, ur_event_handle_t *phEvent) {
  UR_ASSERT(pMem && size > 0, UR_RESULT_ERROR_INVALID_VALUE);
  void *HIPDevicePtr = const_cast<void *>(pMem);
  ur_device_handle_t Device = hQueue->getDevice();

#if HIP_VERSION_MAJOR >= 5
  // NOTE: The hipPointerGetAttribute API is marked as beta, meaning, while this
  // is feature complete, it is still open to changes and outstanding issues.
  size_t PointerRangeSize = 0;
  UR_CHECK_ERROR(hipPointerGetAttribute(
      &PointerRangeSize, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
      static_cast<hipDeviceptr_t>(HIPDevicePtr)));
  UR_ASSERT(size <= PointerRangeSize, UR_RESULT_ERROR_INVALID_SIZE);
#endif

  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(Device);
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
    // hipMemAdvise on a GPU device requires the GPU device to report a non-zero
    // value for hipDeviceAttributeConcurrentManagedAccess. Therefore, ignore
    // the mem advice if concurrent managed memory access is not available.
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

    const auto DeviceID = Device->get();
    if (advice & UR_USM_ADVICE_FLAG_DEFAULT) {
      UR_CHECK_ERROR(
          hipMemAdvise(pMem, size, hipMemAdviseUnsetReadMostly, DeviceID));
      UR_CHECK_ERROR(hipMemAdvise(
          pMem, size, hipMemAdviseUnsetPreferredLocation, DeviceID));
      UR_CHECK_ERROR(
          hipMemAdvise(pMem, size, hipMemAdviseUnsetAccessedBy, DeviceID));
    } else {
      Result = setHipMemAdvise(HIPDevicePtr, size, advice, DeviceID);
      // UR_RESULT_ERROR_INVALID_ENUMERATION is returned when using a valid but
      // currently unmapped advice arguments as not supported by this platform.
      // Therefore, warn the user instead of throwing and aborting the runtime.
      if (Result == UR_RESULT_ERROR_INVALID_ENUMERATION) {
        releaseEvent();
        setErrorMessage("mem_advise is ignored as the advice argument is not "
                        "supported by this device",
                        UR_RESULT_SUCCESS);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
    }

    releaseEvent();
  } catch (ur_result_t err) {
    Result = err;
  } catch (...) {
    Result = UR_RESULT_ERROR_UNKNOWN;
  }

  return Result;
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
    ScopedContext Active(hQueue->getDevice());
    hipStream_t HIPStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                               phEventWaitList);

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_MEMCPY_2D, hQueue, HIPStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    // There is an issue with hipMemcpy2D* when hipMemcpyDefault is used, which
    // makes the HIP runtime not correctly derive the copy kind (direction) for
    // the copies since ROCm 5.6.0+. See: https://github.com/ROCm/clr/issues/40
    // TODO: Add maximum HIP_VERSION when bug has been fixed.
#if HIP_VERSION >= 50600000
    hipPointerAttribute_t srcAttribs{};
    hipPointerAttribute_t dstAttribs{};

    bool srcIsSystemAlloc{false};
    bool dstIsSystemAlloc{false};

    hipError_t hipRes{};
    // hipErrorInvalidValue returned from hipPointerGetAttributes for a non-null
    // pointer refers to an OS-allocation, hence pageable host memory. However,
    // this means we cannot rely on the attributes result, hence we mark system
    // pageable memory allocation manually as host memory. The HIP runtime can
    // handle the registering/unregistering of the memory as long as the right
    // copy-kind (direction) is provided to hipMemcpy2DAsync for this case.
    hipRes = hipPointerGetAttributes(&srcAttribs, (const void *)pSrc);
    if (hipRes == hipErrorInvalidValue && pSrc)
      srcIsSystemAlloc = true;
    hipRes = hipPointerGetAttributes(&dstAttribs, (const void *)pDst);
    if (hipRes == hipErrorInvalidValue && pDst)
      dstIsSystemAlloc = true;

    const unsigned int srcMemType{srcAttribs.type};
    const unsigned int dstMemType{dstAttribs.type};

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
    Result = Err;
  }

  return Result;
}

namespace {

enum class GlobalVariableCopy { Read, Write };

ur_result_t deviceGlobalCopyHelper(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blocking, size_t count, size_t offset, void *ptr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, GlobalVariableCopy CopyType) {
  // Since HIP requires a the global variable to be referenced by name, we use
  // metadata to find the correct name to access it by.
  auto DeviceGlobalNameIt = hProgram->GlobalIDMD.find(name);
  if (DeviceGlobalNameIt == hProgram->GlobalIDMD.end())
    return UR_RESULT_ERROR_INVALID_VALUE;
  std::string DeviceGlobalName = DeviceGlobalNameIt->second;

  try {
    hipDeviceptr_t DeviceGlobal = 0;
    size_t DeviceGlobalSize = 0;
    UR_CHECK_ERROR(hipModuleGetGlobal(&DeviceGlobal, &DeviceGlobalSize,
                                      hProgram->get(),
                                      DeviceGlobalName.c_str()));

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
