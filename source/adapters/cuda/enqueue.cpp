//===--------- enqueue.cpp - CUDA Adapter ---------------------------------===//
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

#include <cmath>
#include <cuda.h>

ur_result_t enqueueEventsWait(ur_queue_handle_t CommandQueue, CUstream Stream,
                              uint32_t NumEventsInWaitList,
                              const ur_event_handle_t *EventWaitList) {
  UR_ASSERT(EventWaitList, UR_RESULT_SUCCESS);

  try {
    ScopedContext Active(CommandQueue->getContext());

    auto Result = forLatestEvents(
        EventWaitList, NumEventsInWaitList,
        [Stream](ur_event_handle_t Event) -> ur_result_t {
          if (Event->getStream() == Stream) {
            return UR_RESULT_SUCCESS;
          } else {
            UR_CHECK_ERROR(cuStreamWaitEvent(Stream, Event->get(), 0));
            return UR_RESULT_SUCCESS;
          }
        });
    return Result;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

template <typename PtrT>
void getUSMHostOrDevicePtr(PtrT USMPtr, CUmemorytype *OutMemType,
                           CUdeviceptr *OutDevPtr, PtrT *OutHostPtr) {
  // do not throw if cuPointerGetAttribute returns CUDA_ERROR_INVALID_VALUE
  // checks with PI_CHECK_ERROR are not suggested
  CUresult Ret = cuPointerGetAttribute(
      OutMemType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)USMPtr);
  // ARRAY, UNIFIED types are not supported!
  assert(*OutMemType != CU_MEMORYTYPE_ARRAY &&
         *OutMemType != CU_MEMORYTYPE_UNIFIED);

  // pointer not known to the CUDA subsystem (possibly a system allocated ptr)
  if (Ret == CUDA_ERROR_INVALID_VALUE) {
    *OutMemType = CU_MEMORYTYPE_HOST;
    *OutDevPtr = 0;
    *OutHostPtr = USMPtr;

    // todo: resets the above "non-stick" error
  } else if (Ret == CUDA_SUCCESS) {
    *OutDevPtr = (*OutMemType == CU_MEMORYTYPE_DEVICE)
                     ? reinterpret_cast<CUdeviceptr>(USMPtr)
                     : 0;
    *OutHostPtr = (*OutMemType == CU_MEMORYTYPE_HOST) ? USMPtr : nullptr;
  } else {
    UR_CHECK_ERROR(Ret);
  }
}

ur_result_t setCuMemAdvise(CUdeviceptr DevPtr, size_t Size,
                           ur_usm_advice_flags_t URAdviceFlags,
                           CUdevice Device) {
  std::unordered_map<ur_usm_advice_flags_t, CUmem_advise>
      URToCUMemAdviseDeviceFlagsMap = {
          {UR_USM_ADVICE_FLAG_SET_READ_MOSTLY, CU_MEM_ADVISE_SET_READ_MOSTLY},
          {UR_USM_ADVICE_FLAG_CLEAR_READ_MOSTLY,
           CU_MEM_ADVISE_UNSET_READ_MOSTLY},
          {UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION,
           CU_MEM_ADVISE_SET_PREFERRED_LOCATION},
          {UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION,
           CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION},
          {UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE,
           CU_MEM_ADVISE_SET_ACCESSED_BY},
          {UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE,
           CU_MEM_ADVISE_UNSET_ACCESSED_BY},
      };
  for (auto &FlagPair : URToCUMemAdviseDeviceFlagsMap) {
    if (URAdviceFlags & FlagPair.first) {
      UR_CHECK_ERROR(cuMemAdvise(DevPtr, Size, FlagPair.second, Device));
    }
  }

  std::unordered_map<ur_usm_advice_flags_t, CUmem_advise>
      URToCUMemAdviseHostFlagsMap = {
          {UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION_HOST,
           CU_MEM_ADVISE_SET_PREFERRED_LOCATION},
          {UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION_HOST,
           CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION},
          {UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_HOST,
           CU_MEM_ADVISE_SET_ACCESSED_BY},
          {UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_HOST,
           CU_MEM_ADVISE_UNSET_ACCESSED_BY},
      };

  for (auto &FlagPair : URToCUMemAdviseHostFlagsMap) {
    if (URAdviceFlags & FlagPair.first) {
      UR_CHECK_ERROR(cuMemAdvise(DevPtr, Size, FlagPair.second, CU_DEVICE_CPU));
    }
  }

  std::array<ur_usm_advice_flags_t, 6> UnmappedMemAdviceFlags = {
      UR_USM_ADVICE_FLAG_SET_NON_ATOMIC_MOSTLY,
      UR_USM_ADVICE_FLAG_CLEAR_NON_ATOMIC_MOSTLY,
      UR_USM_ADVICE_FLAG_BIAS_CACHED,
      UR_USM_ADVICE_FLAG_BIAS_UNCACHED,
      UR_USM_ADVICE_FLAG_SET_NON_COHERENT_MEMORY,
      UR_USM_ADVICE_FLAG_CLEAR_NON_COHERENT_MEMORY};

  for (auto &UnmappedFlag : UnmappedMemAdviceFlags) {
    if (URAdviceFlags & UnmappedFlag) {
      setErrorMessage("Memory advice ignored because the CUDA backend does not "
                      "support some of the specified flags",
                      UR_RESULT_SUCCESS);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }
  }

  return UR_RESULT_SUCCESS;
}

// Determine local work sizes that result in uniform work groups.
// The default threadsPerBlock only require handling the first work_dim
// dimension.
void guessLocalWorkSize(ur_device_handle_t Device, size_t *ThreadsPerBlock,
                        const size_t *GlobalWorkSize, const uint32_t WorkDim,
                        const size_t MaxThreadsPerBlock[3],
                        ur_kernel_handle_t Kernel, uint32_t LocalSize) {
  assert(ThreadsPerBlock != nullptr);
  assert(GlobalWorkSize != nullptr);
  assert(Kernel != nullptr);
  int MinGrid, MaxBlockSize;
  size_t MaxBlockDim[3];

  // The below assumes a three dimensional range but this is not guaranteed by
  // UR.
  size_t GlobalSizeNormalized[3] = {1, 1, 1};
  for (uint32_t i = 0; i < WorkDim; i++) {
    GlobalSizeNormalized[i] = GlobalWorkSize[i];
  }

  MaxBlockDim[1] = Device->getMaxBlockDimY();
  MaxBlockDim[2] = Device->getMaxBlockDimZ();

  UR_CHECK_ERROR(
      cuOccupancyMaxPotentialBlockSize(&MinGrid, &MaxBlockSize, Kernel->get(),
                                       NULL, LocalSize, MaxThreadsPerBlock[0]));

  ThreadsPerBlock[2] = std::min(GlobalSizeNormalized[2], MaxBlockDim[2]);
  ThreadsPerBlock[1] =
      std::min(GlobalSizeNormalized[1],
               std::min(MaxBlockSize / ThreadsPerBlock[2], MaxBlockDim[1]));
  MaxBlockDim[0] = MaxBlockSize / (ThreadsPerBlock[1] * ThreadsPerBlock[2]);
  ThreadsPerBlock[0] = std::min(
      MaxThreadsPerBlock[0], std::min(GlobalSizeNormalized[0], MaxBlockDim[0]));

  static auto IsPowerOf2 = [](size_t Value) -> bool {
    return Value && !(Value & (Value - 1));
  };

  // Find a local work group size that is a divisor of the global
  // work group size to produce uniform work groups.
  // Additionally, for best compute utilisation, the local size has
  // to be a power of two.
  while (0u != (GlobalSizeNormalized[0] % ThreadsPerBlock[0]) ||
         !IsPowerOf2(ThreadsPerBlock[0])) {
    --ThreadsPerBlock[0];
  }
}

// Helper to verify out-of-registers case (exceeded block max registers).
// If the kernel requires a number of registers for the entire thread
// block exceeds the hardware limitations, then the cuLaunchKernel call
// will fail to launch with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES error.
bool hasExceededMaxRegistersPerBlock(ur_device_handle_t Device,
                                     ur_kernel_handle_t Kernel,
                                     size_t BlockSize) {
  return BlockSize * Kernel->getRegsPerThread() > Device->getMaxRegsPerBlock();
}

// Helper to compute kernel parameters from workload
// dimensions.
// @param [in]  Context handler to the target Context
// @param [in]  Device handler to the target Device
// @param [in]  WorkDim workload dimension
// @param [in]  GlobalWorkOffset pointer workload global offsets
// @param [in]  LocalWorkOffset pointer workload local offsets
// @param [inout] Kernel handler to the kernel
// @param [inout] CuFunc handler to the cuda function attached to the kernel
// @param [out] ThreadsPerBlock Number of threads per block we should run
// @param [out] BlocksPerGrid Number of blocks per grid we should run
ur_result_t
setKernelParams(const ur_context_handle_t Context,
                const ur_device_handle_t Device, const uint32_t WorkDim,
                const size_t *GlobalWorkOffset, const size_t *GlobalWorkSize,
                const size_t *LocalWorkSize, ur_kernel_handle_t &Kernel,
                CUfunction &CuFunc, size_t (&ThreadsPerBlock)[3],
                size_t (&BlocksPerGrid)[3]) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  size_t MaxWorkGroupSize = 0u;
  size_t MaxThreadsPerBlock[3] = {};
  bool ProvidedLocalWorkGroupSize = LocalWorkSize != nullptr;
  uint32_t LocalSize = Kernel->getLocalSize();

  try {
    // Set the active context here as guessLocalWorkSize needs an active context
    ScopedContext Active(Context);
    {
      size_t *ReqdThreadsPerBlock = Kernel->ReqdThreadsPerBlock;
      MaxWorkGroupSize = Device->getMaxWorkGroupSize();
      Device->getMaxWorkItemSizes(sizeof(MaxThreadsPerBlock),
                                  MaxThreadsPerBlock);

      if (ProvidedLocalWorkGroupSize) {
        auto IsValid = [&](int Dim) {
          if (ReqdThreadsPerBlock[Dim] != 0 &&
              LocalWorkSize[Dim] != ReqdThreadsPerBlock[Dim])
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;

          if (LocalWorkSize[Dim] > MaxThreadsPerBlock[Dim])
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
          // Checks that local work sizes are a divisor of the global work sizes
          // which includes that the local work sizes are neither larger than
          // the global work sizes and not 0.
          if (0u == LocalWorkSize[Dim])
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
          if (0u != (GlobalWorkSize[Dim] % LocalWorkSize[Dim]))
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
          ThreadsPerBlock[Dim] = LocalWorkSize[Dim];
          return UR_RESULT_SUCCESS;
        };

        size_t KernelLocalWorkGroupSize = 0;
        for (size_t Dim = 0; Dim < WorkDim; Dim++) {
          auto Err = IsValid(Dim);
          if (Err != UR_RESULT_SUCCESS)
            return Err;
          // If no error then sum the total local work size per dim.
          KernelLocalWorkGroupSize += LocalWorkSize[Dim];
        }

        if (hasExceededMaxRegistersPerBlock(Device, Kernel,
                                            KernelLocalWorkGroupSize)) {
          return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
        }
      } else {
        guessLocalWorkSize(Device, ThreadsPerBlock, GlobalWorkSize, WorkDim,
                           MaxThreadsPerBlock, Kernel, LocalSize);
      }
    }

    if (MaxWorkGroupSize <
        ThreadsPerBlock[0] * ThreadsPerBlock[1] * ThreadsPerBlock[2]) {
      return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
    }

    for (size_t i = 0; i < WorkDim; i++) {
      BlocksPerGrid[i] =
          (GlobalWorkSize[i] + ThreadsPerBlock[i] - 1) / ThreadsPerBlock[i];
    }

    // Set the implicit global offset parameter if kernel has offset variant
    if (Kernel->get_with_offset_parameter()) {
      std::uint32_t CudaImplicitOffset[3] = {0, 0, 0};
      if (GlobalWorkOffset) {
        for (size_t i = 0; i < WorkDim; i++) {
          CudaImplicitOffset[i] =
              static_cast<std::uint32_t>(GlobalWorkOffset[i]);
          if (GlobalWorkOffset[i] != 0) {
            CuFunc = Kernel->get_with_offset_parameter();
          }
        }
      }
      Kernel->setImplicitOffsetArg(sizeof(CudaImplicitOffset),
                                   CudaImplicitOffset);
    }

    auto Device = Context->getDevice();
    if (LocalSize > static_cast<uint32_t>(Device->getMaxCapacityLocalMem())) {
      setErrorMessage("Excessive allocation of local memory on the device",
                      UR_RESULT_ERROR_ADAPTER_SPECIFIC);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }

    if (Device->maxLocalMemSizeChosen()) {
      // Set up local memory requirements for kernel.
      if (Device->getMaxChosenLocalMem() < 0) {
        bool EnvVarHasURPrefix =
            std::getenv("UR_CUDA_MAX_LOCAL_MEM_SIZE") != nullptr;
        setErrorMessage(EnvVarHasURPrefix ? "Invalid value specified for "
                                            "UR_CUDA_MAX_LOCAL_MEM_SIZE"
                                          : "Invalid value specified for "
                                            "SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE",
                        UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      if (LocalSize > static_cast<uint32_t>(Device->getMaxChosenLocalMem())) {
        bool EnvVarHasURPrefix =
            std::getenv("UR_CUDA_MAX_LOCAL_MEM_SIZE") != nullptr;
        setErrorMessage(
            EnvVarHasURPrefix
                ? "Local memory for kernel exceeds the amount requested using "
                  "UR_CUDA_MAX_LOCAL_MEM_SIZE. Try increasing the value of "
                  "UR_CUDA_MAX_LOCAL_MEM_SIZE."
                : "Local memory for kernel exceeds the amount requested using "
                  "SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE. Try increasing the the "
                  "value of SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE.",
            UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      UR_CHECK_ERROR(cuFuncSetAttribute(
          CuFunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          Device->getMaxChosenLocalMem()));

    } else {
      UR_CHECK_ERROR(cuFuncSetAttribute(
          CuFunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, LocalSize));
    }

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

/// Enqueues a wait on the given CUstream for all specified events (See
/// \ref enqueueEventWaitWithBarrier.) If the events list is empty, the enqueued
/// wait will wait on all previous events in the queue.
///
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // This function makes one stream work on the previous work (or work
  // represented by input events) and then all future work waits on that stream.
  try {
    ScopedContext Active(hQueue->getContext());
    uint32_t StreamToken;
    ur_stream_guard_ Guard;
    CUstream CuStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    {
      std::lock_guard<std::mutex> GuardBarrier(hQueue->BarrierMutex);
      if (hQueue->BarrierEvent == nullptr) {
        UR_CHECK_ERROR(
            cuEventCreate(&hQueue->BarrierEvent, CU_EVENT_DISABLE_TIMING));
      }
      if (numEventsInWaitList == 0) { //  wait on all work
        if (hQueue->BarrierTmpEvent == nullptr) {
          UR_CHECK_ERROR(
              cuEventCreate(&hQueue->BarrierTmpEvent, CU_EVENT_DISABLE_TIMING));
        }
        hQueue->syncStreams(
            [CuStream, TmpEvent = hQueue->BarrierTmpEvent](CUstream s) {
              if (CuStream != s) {
                // record a new CUDA event on every stream and make one stream
                // wait for these events
                UR_CHECK_ERROR(cuEventRecord(TmpEvent, s));
                UR_CHECK_ERROR(cuStreamWaitEvent(CuStream, TmpEvent, 0));
              }
            });
      } else { // wait just on given events
        forLatestEvents(phEventWaitList, numEventsInWaitList,
                        [CuStream](ur_event_handle_t Event) -> ur_result_t {
                          if (Event->getQueue()->hasBeenSynchronized(
                                  Event->getComputeStreamToken())) {
                            return UR_RESULT_SUCCESS;
                          } else {
                            UR_CHECK_ERROR(
                                cuStreamWaitEvent(CuStream, Event->get(), 0));
                            return UR_RESULT_SUCCESS;
                          }
                        });
      }

      UR_CHECK_ERROR(cuEventRecord(hQueue->BarrierEvent, CuStream));
      for (unsigned int i = 0; i < hQueue->ComputeAppliedBarrier.size(); i++) {
        hQueue->ComputeAppliedBarrier[i] = false;
      }
      for (unsigned int i = 0; i < hQueue->TransferAppliedBarrier.size(); i++) {
        hQueue->TransferAppliedBarrier[i] = false;
      }
    }

    if (phEvent) {
      *phEvent = ur_event_handle_t_::makeNative(
          UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, hQueue, CuStream, StreamToken);
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

/// Enqueues a wait on the given CUstream for all events.
/// See \ref enqueueEventWait
/// TODO: Add support for multiple streams once the Event class is properly
/// refactored.
///
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // Preconditions
  UR_ASSERT(hQueue->getContext() == hKernel->getContext(),
            UR_RESULT_ERROR_INVALID_KERNEL);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  if (*pGlobalWorkSize == 0) {
    return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                          phEventWaitList, phEvent);
  }

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  size_t ThreadsPerBlock[3] = {32u, 1u, 1u};
  size_t BlocksPerGrid[3] = {1u, 1u, 1u};

  uint32_t LocalSize = hKernel->getLocalSize();
  ur_result_t Result = UR_RESULT_SUCCESS;
  CUfunction CuFunc = hKernel->get();

  Result = setKernelParams(hQueue->getContext(), hQueue->Device, workDim,
                           pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
                           hKernel, CuFunc, ThreadsPerBlock, BlocksPerGrid);
  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

    uint32_t StreamToken;
    ur_stream_guard_ Guard;
    CUstream CuStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_KERNEL_LAUNCH, hQueue, CuStream, StreamToken));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    auto &ArgIndices = hKernel->getArgIndices();
    UR_CHECK_ERROR(cuLaunchKernel(
        CuFunc, BlocksPerGrid[0], BlocksPerGrid[1], BlocksPerGrid[2],
        ThreadsPerBlock[0], ThreadsPerBlock[1], ThreadsPerBlock[2], LocalSize,
        CuStream, const_cast<void **>(ArgIndices.data()), nullptr));

    if (LocalSize != 0)
      hKernel->clearLocalSize();

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
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

/// Set parameters for general 3D memory copy.
/// If the source and/or destination is on the device, SrcPtr and/or DstPtr
/// must be a pointer to a CUdeviceptr
void setCopyRectParams(ur_rect_region_t region, const void *SrcPtr,
                       const CUmemorytype_enum SrcType,
                       ur_rect_offset_t src_offset, size_t src_row_pitch,
                       size_t src_slice_pitch, void *DstPtr,
                       const CUmemorytype_enum DstType,
                       ur_rect_offset_t dst_offset, size_t dst_row_pitch,
                       size_t dst_slice_pitch, CUDA_MEMCPY3D &params) {
  src_row_pitch =
      (!src_row_pitch) ? region.width + src_offset.x : src_row_pitch;
  src_slice_pitch = (!src_slice_pitch)
                        ? ((region.height + src_offset.y) * src_row_pitch)
                        : src_slice_pitch;
  dst_row_pitch =
      (!dst_row_pitch) ? region.width + dst_offset.x : dst_row_pitch;
  dst_slice_pitch = (!dst_slice_pitch)
                        ? ((region.height + dst_offset.y) * dst_row_pitch)
                        : dst_slice_pitch;

  params.WidthInBytes = region.width;
  params.Height = region.height;
  params.Depth = region.depth;

  params.srcMemoryType = SrcType;
  params.srcDevice = SrcType == CU_MEMORYTYPE_DEVICE
                         ? *static_cast<const CUdeviceptr *>(SrcPtr)
                         : 0;
  params.srcHost = SrcType == CU_MEMORYTYPE_HOST ? SrcPtr : nullptr;
  params.srcXInBytes = src_offset.x;
  params.srcY = src_offset.y;
  params.srcZ = src_offset.z;
  params.srcPitch = src_row_pitch;
  params.srcHeight = src_slice_pitch / src_row_pitch;

  params.dstMemoryType = DstType;
  params.dstDevice =
      DstType == CU_MEMORYTYPE_DEVICE ? *static_cast<CUdeviceptr *>(DstPtr) : 0;
  params.dstHost = DstType == CU_MEMORYTYPE_HOST ? DstPtr : nullptr;
  params.dstXInBytes = dst_offset.x;
  params.dstY = dst_offset.y;
  params.dstZ = dst_offset.z;
  params.dstPitch = dst_row_pitch;
  params.dstHeight = dst_slice_pitch / dst_row_pitch;
}

/// General 3D memory copy operation.
/// This function requires the corresponding CUDA context to be at the top of
/// the context stack
/// If the source and/or destination is on the device, SrcPtr and/or DstPtr
/// must be a pointer to a CUdeviceptr
static ur_result_t commonEnqueueMemBufferCopyRect(
    CUstream cu_stream, ur_rect_region_t region, const void *SrcPtr,
    const CUmemorytype_enum SrcType, ur_rect_offset_t src_offset,
    size_t src_row_pitch, size_t src_slice_pitch, void *DstPtr,
    const CUmemorytype_enum DstType, ur_rect_offset_t dst_offset,
    size_t dst_row_pitch, size_t dst_slice_pitch) {
  UR_ASSERT(SrcType == CU_MEMORYTYPE_DEVICE || SrcType == CU_MEMORYTYPE_HOST,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(DstType == CU_MEMORYTYPE_DEVICE || DstType == CU_MEMORYTYPE_HOST,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  CUDA_MEMCPY3D params = {};

  setCopyRectParams(region, SrcPtr, SrcType, src_offset, src_row_pitch,
                    src_slice_pitch, DstPtr, DstType, dst_offset, dst_row_pitch,
                    dst_slice_pitch, params);

  UR_CHECK_ERROR(cuMemcpy3DAsync(&params, cu_stream));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  CUdeviceptr DevPtr = std::get<BufferMem>(hBuffer->Mem).get();
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();

    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, CuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Result = commonEnqueueMemBufferCopyRect(
        CuStream, region, &DevPtr, CU_MEMORYTYPE_DEVICE, bufferOrigin,
        bufferRowPitch, bufferSlicePitch, pDst, CU_MEMORYTYPE_HOST, hostOrigin,
        hostRowPitch, hostSlicePitch);

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingRead) {
      UR_CHECK_ERROR(cuStreamSynchronize(CuStream));
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
  CUdeviceptr DevPtr = std::get<BufferMem>(hBuffer->Mem).get();
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext active(hQueue->getContext());
    CUstream cuStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, cuStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_WRITE_RECT, hQueue, cuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Result = commonEnqueueMemBufferCopyRect(
        cuStream, region, pSrc, CU_MEMORYTYPE_HOST, hostOrigin, hostRowPitch,
        hostSlicePitch, &DevPtr, CU_MEMORYTYPE_DEVICE, bufferOrigin,
        bufferRowPitch, bufferSlicePitch);

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingWrite) {
      UR_CHECK_ERROR(cuStreamSynchronize(cuStream));
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
  UR_ASSERT(size + dstOffset <= std::get<BufferMem>(hBufferDst->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(size + srcOffset <= std::get<BufferMem>(hBufferSrc->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());
    ur_result_t Result = UR_RESULT_SUCCESS;

    auto Stream = hQueue->getNextTransferStream();
    Result =
        enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY, hQueue, Stream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    auto Src = std::get<BufferMem>(hBufferSrc->Mem).get() + srcOffset;
    auto Dst = std::get<BufferMem>(hBufferDst->Mem).get() + dstOffset;

    UR_CHECK_ERROR(cuMemcpyDtoDAsync(Dst, Src, size, Stream));

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
  CUdeviceptr SrcPtr = std::get<BufferMem>(hBufferSrc->Mem).get();
  CUdeviceptr DstPtr = std::get<BufferMem>(hBufferDst->Mem).get();
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY_RECT, hQueue, CuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    Result = commonEnqueueMemBufferCopyRect(
        CuStream, region, &SrcPtr, CU_MEMORYTYPE_DEVICE, srcOrigin, srcRowPitch,
        srcSlicePitch, &DstPtr, CU_MEMORYTYPE_DEVICE, dstOrigin, dstRowPitch,
        dstSlicePitch);

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t err) {
    Result = err;
  }
  return Result;
}

// CUDA has no memset functions that allow setting values more than 4 bytes. UR
// API lets you pass an arbitrary "pattern" to the buffer fill, which can be
// more than 4 bytes. We must break up the pattern into 1 byte values, and set
// the buffer using multiple strided calls.  The first 4 patterns are set using
// cuMemsetD32Async then all subsequent 1 byte patterns are set using
// cuMemset2DAsync which is called for each pattern.
ur_result_t commonMemSetLargePattern(CUstream Stream, uint32_t PatternSize,
                                     size_t Size, const void *pPattern,
                                     CUdeviceptr Ptr) {
  // Calculate the number of patterns, stride, number of times the pattern
  // needs to be applied, and the number of times the first 32 bit pattern
  // needs to be applied.
  auto NumberOfSteps = PatternSize / sizeof(uint8_t);
  auto Pitch = NumberOfSteps * sizeof(uint8_t);
  auto Height = Size / NumberOfSteps;
  auto Count32 = Size / sizeof(uint32_t);

  // Get 4-byte chunk of the pattern and call cuMemsetD32Async
  auto Value = *(static_cast<const uint32_t *>(pPattern));
  UR_CHECK_ERROR(cuMemsetD32Async(Ptr, Value, Count32, Stream));
  for (auto step = 4u; step < NumberOfSteps; ++step) {
    // take 1 byte of the pattern
    Value = *(static_cast<const uint8_t *>(pPattern) + step);

    // offset the pointer to the part of the buffer we want to write to
    auto OffsetPtr = Ptr + (step * sizeof(uint8_t));

    // set all of the pattern chunks
    UR_CHECK_ERROR(cuMemsetD2D8Async(OffsetPtr, Pitch, Value, sizeof(uint8_t),
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

  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());

    auto Stream = hQueue->getNextTransferStream();
    ur_result_t Result =
        enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_FILL, hQueue, Stream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    auto DstDevice = std::get<BufferMem>(hBuffer->Mem).get() + offset;
    auto N = size / patternSize;

    // pattern size in bytes
    switch (patternSize) {
    case 1: {
      auto Value = *static_cast<const uint8_t *>(pPattern);
      UR_CHECK_ERROR(cuMemsetD8Async(DstDevice, Value, N, Stream));
      break;
    }
    case 2: {
      auto Value = *static_cast<const uint16_t *>(pPattern);
      UR_CHECK_ERROR(cuMemsetD16Async(DstDevice, Value, N, Stream));
      break;
    }
    case 4: {
      auto Value = *static_cast<const uint32_t *>(pPattern);
      UR_CHECK_ERROR(cuMemsetD32Async(DstDevice, Value, N, Stream));
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

static size_t imageElementByteSize(CUDA_ARRAY_DESCRIPTOR ArrayDesc) {
  switch (ArrayDesc.Format) {
  case CU_AD_FORMAT_UNSIGNED_INT8:
  case CU_AD_FORMAT_SIGNED_INT8:
    return 1;
  case CU_AD_FORMAT_UNSIGNED_INT16:
  case CU_AD_FORMAT_SIGNED_INT16:
  case CU_AD_FORMAT_HALF:
    return 2;
  case CU_AD_FORMAT_UNSIGNED_INT32:
  case CU_AD_FORMAT_SIGNED_INT32:
  case CU_AD_FORMAT_FLOAT:
    return 4;
  default:
    detail::ur::die("Invalid image format.");
    return 0;
  }
}

/// General ND memory copy operation for images.
/// This function requires the corresponding CUDA context to be at the top of
/// the context stack
/// If the source and/or destination is an array, SrcPtr and/or DstPtr
/// must be a pointer to a CUarray
static ur_result_t commonEnqueueMemImageNDCopy(
    CUstream CuStream, ur_mem_type_t ImgType, const ur_rect_region_t Region,
    const void *SrcPtr, const CUmemorytype_enum SrcType,
    const ur_rect_offset_t SrcOffset, void *DstPtr,
    const CUmemorytype_enum DstType, const ur_rect_offset_t DstOffset) {
  UR_ASSERT(SrcType == CU_MEMORYTYPE_ARRAY || SrcType == CU_MEMORYTYPE_HOST,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(DstType == CU_MEMORYTYPE_ARRAY || DstType == CU_MEMORYTYPE_HOST,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  if (ImgType == UR_MEM_TYPE_IMAGE1D || ImgType == UR_MEM_TYPE_IMAGE2D) {
    CUDA_MEMCPY2D CpyDesc;
    memset(&CpyDesc, 0, sizeof(CpyDesc));
    CpyDesc.srcMemoryType = SrcType;
    if (SrcType == CU_MEMORYTYPE_ARRAY) {
      CpyDesc.srcArray = *static_cast<const CUarray *>(SrcPtr);
      CpyDesc.srcXInBytes = SrcOffset.x;
      CpyDesc.srcY = (ImgType == UR_MEM_TYPE_IMAGE1D) ? 0 : SrcOffset.y;
    } else {
      CpyDesc.srcHost = SrcPtr;
    }
    CpyDesc.dstMemoryType = DstType;
    if (DstType == CU_MEMORYTYPE_ARRAY) {
      CpyDesc.dstArray = *static_cast<CUarray *>(DstPtr);
      CpyDesc.dstXInBytes = DstOffset.x;
      CpyDesc.dstY = (ImgType == UR_MEM_TYPE_IMAGE1D) ? 0 : DstOffset.y;
    } else {
      CpyDesc.dstHost = DstPtr;
    }
    CpyDesc.WidthInBytes = Region.width;
    CpyDesc.Height = (ImgType == UR_MEM_TYPE_IMAGE1D) ? 1 : Region.height;
    UR_CHECK_ERROR(cuMemcpy2DAsync(&CpyDesc, CuStream));
    return UR_RESULT_SUCCESS;
  }
  if (ImgType == UR_MEM_TYPE_IMAGE3D) {
    CUDA_MEMCPY3D CpyDesc;
    memset(&CpyDesc, 0, sizeof(CpyDesc));
    CpyDesc.srcMemoryType = SrcType;
    if (SrcType == CU_MEMORYTYPE_ARRAY) {
      CpyDesc.srcArray = *static_cast<const CUarray *>(SrcPtr);
      CpyDesc.srcXInBytes = SrcOffset.x;
      CpyDesc.srcY = SrcOffset.y;
      CpyDesc.srcZ = SrcOffset.z;
    } else {
      CpyDesc.srcHost = SrcPtr;
    }
    CpyDesc.dstMemoryType = DstType;
    if (DstType == CU_MEMORYTYPE_ARRAY) {
      CpyDesc.dstArray = *static_cast<CUarray *>(DstPtr);
      CpyDesc.dstXInBytes = DstOffset.x;
      CpyDesc.dstY = DstOffset.y;
      CpyDesc.dstZ = DstOffset.z;
    } else {
      CpyDesc.dstHost = DstPtr;
    }
    CpyDesc.WidthInBytes = Region.width;
    CpyDesc.Height = Region.height;
    CpyDesc.Depth = Region.depth;
    UR_CHECK_ERROR(cuMemcpy3DAsync(&CpyDesc, CuStream));
    return UR_RESULT_SUCCESS;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingRead,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = rowPitch;
  std::ignore = slicePitch;

  UR_ASSERT(hImage->MemType == ur_mem_handle_t_::Type::Surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    CUarray Array = std::get<SurfaceMem>(hImage->Mem).getArray();

    CUDA_ARRAY_DESCRIPTOR ArrayDesc;
    UR_CHECK_ERROR(cuArrayGetDescriptor(&ArrayDesc, Array));

    int ElementByteSize = imageElementByteSize(ArrayDesc);

    size_t ByteOffsetX = origin.x * ElementByteSize * ArrayDesc.NumChannels;
    size_t BytesToCopy = ElementByteSize * ArrayDesc.NumChannels * region.width;

    ur_mem_type_t ImgType = std::get<SurfaceMem>(hImage->Mem).getImageType();

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_IMAGE_READ, hQueue, CuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }
    if (ImgType == UR_MEM_TYPE_IMAGE1D) {
      UR_CHECK_ERROR(
          cuMemcpyAtoHAsync(pDst, Array, ByteOffsetX, BytesToCopy, CuStream));
    } else {
      ur_rect_region_t AdjustedRegion = {BytesToCopy, region.height,
                                         region.depth};
      ur_rect_offset_t SrcOffset = {ByteOffsetX, origin.y, origin.z};

      Result = commonEnqueueMemImageNDCopy(
          CuStream, ImgType, AdjustedRegion, &Array, CU_MEMORYTYPE_ARRAY,
          SrcOffset, pDst, CU_MEMORYTYPE_HOST, ur_rect_offset_t{});
      if (Result != UR_RESULT_SUCCESS) {
        return Result;
      }
    }

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }

    if (blockingRead) {
      UR_CHECK_ERROR(cuStreamSynchronize(CuStream));
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingWrite,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = blockingWrite;
  std::ignore = rowPitch;
  std::ignore = slicePitch;

  UR_ASSERT(hImage->MemType == ur_mem_handle_t_::Type::Surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    CUarray Array = std::get<SurfaceMem>(hImage->Mem).getArray();

    CUDA_ARRAY_DESCRIPTOR ArrayDesc;
    UR_CHECK_ERROR(cuArrayGetDescriptor(&ArrayDesc, Array));

    int ElementByteSize = imageElementByteSize(ArrayDesc);

    size_t ByteOffsetX = origin.x * ElementByteSize * ArrayDesc.NumChannels;
    size_t BytesToCopy = ElementByteSize * ArrayDesc.NumChannels * region.width;

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_IMAGE_WRITE, hQueue, CuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    ur_mem_type_t ImgType = std::get<SurfaceMem>(hImage->Mem).getImageType();
    if (ImgType == UR_MEM_TYPE_IMAGE1D) {
      UR_CHECK_ERROR(
          cuMemcpyHtoAAsync(Array, ByteOffsetX, pSrc, BytesToCopy, CuStream));
    } else {
      ur_rect_region_t AdjustedRegion = {BytesToCopy, region.height,
                                         region.depth};
      ur_rect_offset_t DstOffset = {ByteOffsetX, origin.y, origin.z};

      Result = commonEnqueueMemImageNDCopy(
          CuStream, ImgType, AdjustedRegion, pSrc, CU_MEMORYTYPE_HOST,
          ur_rect_offset_t{}, &Array, CU_MEMORYTYPE_ARRAY, DstOffset);

      if (Result != UR_RESULT_SUCCESS) {
        return Result;
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

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImageSrc,
    ur_mem_handle_t hImageDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hImageSrc->MemType == ur_mem_handle_t_::Type::Surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hImageDst->MemType == ur_mem_handle_t_::Type::Surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(std::get<SurfaceMem>(hImageSrc->Mem).getImageType() ==
                std::get<SurfaceMem>(hImageDst->Mem).getImageType(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    CUarray SrcArray = std::get<SurfaceMem>(hImageSrc->Mem).getArray();
    CUarray DstArray = std::get<SurfaceMem>(hImageDst->Mem).getArray();

    CUDA_ARRAY_DESCRIPTOR SrcArrayDesc;
    UR_CHECK_ERROR(cuArrayGetDescriptor(&SrcArrayDesc, SrcArray));
    CUDA_ARRAY_DESCRIPTOR DstArrayDesc;
    UR_CHECK_ERROR(cuArrayGetDescriptor(&DstArrayDesc, DstArray));

    UR_ASSERT(SrcArrayDesc.Format == DstArrayDesc.Format,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);
    UR_ASSERT(SrcArrayDesc.NumChannels == DstArrayDesc.NumChannels,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);

    int ElementByteSize = imageElementByteSize(SrcArrayDesc);

    size_t DstByteOffsetX =
        dstOrigin.x * ElementByteSize * SrcArrayDesc.NumChannels;
    size_t SrcByteOffsetX =
        srcOrigin.x * ElementByteSize * DstArrayDesc.NumChannels;
    size_t BytesToCopy =
        ElementByteSize * SrcArrayDesc.NumChannels * region.width;

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_IMAGE_COPY, hQueue, CuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    ur_mem_type_t ImgType = std::get<SurfaceMem>(hImageSrc->Mem).getImageType();

    ur_rect_region_t AdjustedRegion = {BytesToCopy, region.height,
                                       region.depth};
    ur_rect_offset_t SrcOffset = {SrcByteOffsetX, srcOrigin.y, srcOrigin.z};
    ur_rect_offset_t DstOffset = {DstByteOffsetX, dstOrigin.y, dstOrigin.z};

    Result = commonEnqueueMemImageNDCopy(
        CuStream, ImgType, AdjustedRegion, &SrcArray, CU_MEMORYTYPE_ARRAY,
        SrcOffset, &DstArray, CU_MEMORYTYPE_ARRAY, DstOffset);
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

  return Result;
}

/// Implements mapping on the host using a BufferRead operation.
/// Mapped pointers are stored in the pi_mem object.
/// If the buffer uses pinned host memory a pointer to that memory is returned
/// and no read operation is done.
///
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferMap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingMap,
    ur_map_flags_t mapFlags, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent, void **ppRetMap) {
  UR_ASSERT(hBuffer->MemType == ur_mem_handle_t_::Type::Buffer,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(offset + size <= std::get<BufferMem>(hBuffer->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  auto &BufferImpl = std::get<BufferMem>(hBuffer->Mem);
  ur_result_t Result = UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  const bool IsPinned =
      BufferImpl.MemAllocMode == BufferMem::AllocMode::AllocHostPtr;

  // Currently no support for overlapping regions
  if (BufferImpl.getMapPtr() != nullptr) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  // Allocate a pointer in the host to store the mapped information
  auto HostPtr = BufferImpl.mapToPtr(size, offset, mapFlags);
  *ppRetMap = BufferImpl.getMapPtr();
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
    ScopedContext Active(hQueue->getContext());

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
      } catch (ur_result_t Err) {
        Result = Err;
      }
    }
  }

  return Result;
}

/// Implements the unmap from the host, using a BufferWrite operation.
/// Requires the mapped pointer to be already registered in the given memobj.
/// If memobj uses pinned host memory, this will not do a write.
///
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  UR_ASSERT(hMem->MemType == ur_mem_handle_t_::Type::Buffer,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(std::get<BufferMem>(hMem->Mem).getMapPtr() != nullptr,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(std::get<BufferMem>(hMem->Mem).getMapPtr() == pMappedPtr,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  const bool IsPinned = std::get<BufferMem>(hMem->Mem).MemAllocMode ==
                        BufferMem::AllocMode::AllocHostPtr;

  if (!IsPinned &&
      (std::get<BufferMem>(hMem->Mem).getMapFlags() & UR_MAP_FLAG_WRITE)) {
    // Pinned host memory is only on host so it doesn't need to be written to.
    Result = urEnqueueMemBufferWrite(
        hQueue, hMem, true, std::get<BufferMem>(hMem->Mem).getMapOffset(),
        std::get<BufferMem>(hMem->Mem).getMapSize(), pMappedPtr,
        numEventsInWaitList, phEventWaitList, phEvent);
  } else {
    ScopedContext Active(hQueue->getContext());

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
      } catch (ur_result_t Err) {
        Result = Err;
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
    ScopedContext Active(hQueue->getContext());
    uint32_t StreamToken;
    ur_stream_guard_ Guard;
    CUstream CuStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);
    UR_CHECK_ERROR(enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                                     phEventWaitList));
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_FILL, hQueue, CuStream, StreamToken));
      UR_CHECK_ERROR(EventPtr->start());
    }

    auto N = size / patternSize;
    switch (patternSize) {
    case 1:
      UR_CHECK_ERROR(cuMemsetD8Async(
          (CUdeviceptr)ptr, *((const uint8_t *)pPattern) & 0xFF, N, CuStream));
      break;
    case 2:
      UR_CHECK_ERROR(cuMemsetD16Async((CUdeviceptr)ptr,
                                      *((const uint16_t *)pPattern) & 0xFFFF, N,
                                      CuStream));
      break;
    case 4:
      UR_CHECK_ERROR(cuMemsetD32Async(
          (CUdeviceptr)ptr, *((const uint32_t *)pPattern) & 0xFFFFFFFF, N,
          CuStream));
      break;
    default:
      commonMemSetLargePattern(CuStream, patternSize, size, pPattern,
                               (CUdeviceptr)ptr);
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
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_MEMCPY, hQueue, CuStream));
      UR_CHECK_ERROR(EventPtr->start());
    }
    UR_CHECK_ERROR(
        cuMemcpyAsync((CUdeviceptr)pDst, (CUdeviceptr)pSrc, size, CuStream));
    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
    }
    if (blocking) {
      UR_CHECK_ERROR(cuStreamSynchronize(CuStream));
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

  size_t PointerRangeSize = 0;
  UR_CHECK_ERROR(cuPointerGetAttribute(
      &PointerRangeSize, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)pMem));
  UR_ASSERT(size <= PointerRangeSize, UR_RESULT_ERROR_INVALID_SIZE);
  ur_device_handle_t Device = hQueue->getContext()->getDevice();

  // Certain cuda devices and Windows do not have support for some Unified
  // Memory features. cuMemPrefetchAsync requires concurrent memory access
  // for managed memory. Therefore, ignore prefetch hint if concurrent managed
  // memory access is not available.
  if (!getAttribute(Device, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)) {
    setErrorMessage("Prefetch hint ignored as device does not support "
                    "concurrent managed access",
                    UR_RESULT_SUCCESS);
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  unsigned int IsManaged;
  UR_CHECK_ERROR(cuPointerGetAttribute(
      &IsManaged, CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr)pMem));
  if (!IsManaged) {
    setErrorMessage("Prefetch hint ignored as prefetch only works with USM",
                    UR_RESULT_SUCCESS);
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();
    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);
    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY, hQueue, CuStream));
      UR_CHECK_ERROR(EventPtr->start());
    }
    UR_CHECK_ERROR(
        cuMemPrefetchAsync((CUdeviceptr)pMem, size, Device->get(), CuStream));
    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
      *phEvent = EventPtr.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

/// USM: memadvise API to govern behavior of automatic migration mechanisms
UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem, size_t size,
                   ur_usm_advice_flags_t advice, ur_event_handle_t *phEvent) {
  size_t PointerRangeSize = 0;
  UR_CHECK_ERROR(cuPointerGetAttribute(
      &PointerRangeSize, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)pMem));
  UR_ASSERT(size <= PointerRangeSize, UR_RESULT_ERROR_INVALID_SIZE);

  // Certain cuda devices and Windows do not have support for some Unified
  // Memory features. Passing CU_MEM_ADVISE_SET/CLEAR_PREFERRED_LOCATION and
  // to cuMemAdvise on a GPU device requires the GPU device to report a non-zero
  // value for CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Therfore, ignore
  // memory advise if concurrent managed memory access is not available.
  if ((advice & UR_USM_ADVICE_FLAG_SET_PREFERRED_LOCATION) ||
      (advice & UR_USM_ADVICE_FLAG_CLEAR_PREFERRED_LOCATION) ||
      (advice & UR_USM_ADVICE_FLAG_SET_ACCESSED_BY_DEVICE) ||
      (advice & UR_USM_ADVICE_FLAG_CLEAR_ACCESSED_BY_DEVICE) ||
      (advice & UR_USM_ADVICE_FLAG_DEFAULT)) {
    ur_device_handle_t Device = hQueue->getContext()->getDevice();
    if (!getAttribute(Device, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)) {
      setErrorMessage("Mem advise ignored as device does not support "
                      "concurrent managed access",
                      UR_RESULT_SUCCESS);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }

    // TODO: If ptr points to valid system-allocated pageable memory we should
    // check that the device also has the
    // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS property.
  }

  unsigned int IsManaged;
  UR_CHECK_ERROR(cuPointerGetAttribute(
      &IsManaged, CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr)pMem));
  if (!IsManaged) {
    setErrorMessage(
        "Memory advice ignored as memory advices only works with USM",
        UR_RESULT_SUCCESS);
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  }

  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> EventPtr{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());

    if (phEvent) {
      EventPtr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_USM_ADVISE, hQueue, hQueue->getNextTransferStream()));
      UR_CHECK_ERROR(EventPtr->start());
    }

    if (advice & UR_USM_ADVICE_FLAG_DEFAULT) {
      UR_CHECK_ERROR(cuMemAdvise((CUdeviceptr)pMem, size,
                                 CU_MEM_ADVISE_UNSET_READ_MOSTLY,
                                 hQueue->getContext()->getDevice()->get()));
      UR_CHECK_ERROR(cuMemAdvise((CUdeviceptr)pMem, size,
                                 CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
                                 hQueue->getContext()->getDevice()->get()));
      UR_CHECK_ERROR(cuMemAdvise((CUdeviceptr)pMem, size,
                                 CU_MEM_ADVISE_UNSET_ACCESSED_BY,
                                 hQueue->getContext()->getDevice()->get()));
    } else {
      Result = setCuMemAdvise((CUdeviceptr)pMem, size, advice,
                              hQueue->getContext()->getDevice()->get());
    }

    if (phEvent) {
      UR_CHECK_ERROR(EventPtr->record());
      *phEvent = EventPtr.release();
    }
  } catch (ur_result_t err) {
    Result = err;
  } catch (...) {
    Result = UR_RESULT_ERROR_UNKNOWN;
  }
  return Result;
}

// TODO: Implement this. Remember to return true for
//       PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT when it is implemented.
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t, void *, size_t, size_t, const void *, size_t, size_t,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, size_t dstPitch,
    const void *pSrc, size_t srcPitch, size_t width, size_t height,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t result = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hQueue->getContext());
    CUstream cuStream = hQueue->getNextTransferStream();
    result = enqueueEventsWait(hQueue, cuStream, numEventsInWaitList,
                               phEventWaitList);

    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_COPY_RECT, hQueue, cuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    // Determine the direction of copy using cuPointerGetAttribute
    // for both the SrcPtr and DstPtr
    CUDA_MEMCPY2D CpyDesc = {};
    memset(&CpyDesc, 0, sizeof(CpyDesc));

    getUSMHostOrDevicePtr(pSrc, &CpyDesc.srcMemoryType, &CpyDesc.srcDevice,
                          &CpyDesc.srcHost);
    getUSMHostOrDevicePtr(pDst, &CpyDesc.dstMemoryType, &CpyDesc.dstDevice,
                          &CpyDesc.dstHost);

    CpyDesc.dstPitch = dstPitch;
    CpyDesc.srcPitch = srcPitch;
    CpyDesc.WidthInBytes = width;
    CpyDesc.Height = height;

    UR_CHECK_ERROR(cuMemcpy2DAsync(&CpyDesc, cuStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
    if (blocking) {
      UR_CHECK_ERROR(cuStreamSynchronize(cuStream));
    }
  } catch (ur_result_t err) {
    result = err;
  }
  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(!hBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(offset + size <= std::get<BufferMem>(hBuffer->Mem).Size,
            UR_RESULT_ERROR_INVALID_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  CUdeviceptr DevPtr = std::get<BufferMem>(hBuffer->Mem).get();
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();

    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_READ, hQueue, CuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(cuMemcpyDtoHAsync(pDst, DevPtr + offset, size, CuStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingRead) {
      UR_CHECK_ERROR(cuStreamSynchronize(CuStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }

  } catch (ur_result_t Err) {
    Result = Err;
  }

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(!hBuffer->isImage(), UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(offset + size <= std::get<BufferMem>(hBuffer->Mem).Size,
            UR_RESULT_ERROR_INVALID_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  CUdeviceptr DevPtr = std::get<BufferMem>(hBuffer->Mem).get();
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream CuStream = hQueue->getNextTransferStream();

    Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_MEM_BUFFER_WRITE, hQueue, CuStream));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    UR_CHECK_ERROR(cuMemcpyHtoDAsync(DevPtr + offset, pSrc, size, CuStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
    }

    if (blockingWrite) {
      UR_CHECK_ERROR(cuStreamSynchronize(CuStream));
    }

    if (phEvent) {
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // Since CUDA requires a the global variable to be referenced by name, we use
  // metadata to find the correct name to access it by.
  auto DeviceGlobalNameIt = hProgram->GlobalIDMD.find(name);
  if (DeviceGlobalNameIt == hProgram->GlobalIDMD.end())
    return UR_RESULT_ERROR_INVALID_VALUE;
  std::string DeviceGlobalName = DeviceGlobalNameIt->second;

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    CUdeviceptr DeviceGlobal = 0;
    size_t DeviceGlobalSize = 0;
    UR_CHECK_ERROR(cuModuleGetGlobal(&DeviceGlobal, &DeviceGlobalSize,
                                     hProgram->get(),
                                     DeviceGlobalName.c_str()));

    if (offset + count > DeviceGlobalSize)
      return UR_RESULT_ERROR_INVALID_VALUE;

    return urEnqueueUSMMemcpy(
        hQueue, blockingWrite, reinterpret_cast<void *>(DeviceGlobal + offset),
        pSrc, count, numEventsInWaitList, phEventWaitList, phEvent);
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // Since CUDA requires a the global variable to be referenced by name, we use
  // metadata to find the correct name to access it by.
  auto DeviceGlobalNameIt = hProgram->GlobalIDMD.find(name);
  if (DeviceGlobalNameIt == hProgram->GlobalIDMD.end())
    return UR_RESULT_ERROR_INVALID_VALUE;
  std::string DeviceGlobalName = DeviceGlobalNameIt->second;

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    CUdeviceptr DeviceGlobal = 0;
    size_t DeviceGlobalSize = 0;
    UR_CHECK_ERROR(cuModuleGetGlobal(&DeviceGlobal, &DeviceGlobalSize,
                                     hProgram->get(),
                                     DeviceGlobalName.c_str()));

    if (offset + count > DeviceGlobalSize)
      return UR_RESULT_ERROR_INVALID_VALUE;

    return urEnqueueUSMMemcpy(
        hQueue, blockingRead, pDst,
        reinterpret_cast<const void *>(DeviceGlobal + offset), count,
        numEventsInWaitList, phEventWaitList, phEvent);
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

/// Host Pipes
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pDst, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  (void)hQueue;
  (void)hProgram;
  (void)pipe_symbol;
  (void)blocking;
  (void)pDst;
  (void)size;
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  (void)phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  (void)hQueue;
  (void)hProgram;
  (void)pipe_symbol;
  (void)blocking;
  (void)pSrc;
  (void)size;
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  (void)phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
