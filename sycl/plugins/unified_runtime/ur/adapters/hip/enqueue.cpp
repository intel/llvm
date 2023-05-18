//===--------- enqueue.cpp - HIP Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"

namespace {

static size_t imageElementByteSize(hipArray_Format array_format) {
  switch (array_format) {
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
    sycl::detail::ur::die("Invalid image format.");
  }
  return 0;
}

ur_result_t enqueueEventsWait(ur_queue_handle_t command_queue,
                              hipStream_t stream,
                              uint32_t num_events_in_wait_list,
                              const ur_event_handle_t *event_wait_list) {
  if (!event_wait_list) {
    return UR_RESULT_SUCCESS;
  }
  try {
    ScopedContext active(command_queue->get_context());

    auto result = forLatestEvents(
        event_wait_list, num_events_in_wait_list,
        [stream](ur_event_handle_t event) -> ur_result_t {
          if (event->get_stream() == stream) {
            return UR_RESULT_SUCCESS;
          } else {
            return UR_CHECK_ERROR(hipStreamWaitEvent(stream, event->get(), 0));
          }
        });

    if (result != UR_RESULT_SUCCESS) {
      return result;
    }
    return UR_RESULT_SUCCESS;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

void simpleGuessLocalWorkSize(size_t *threadsPerBlock,
                              const size_t *global_work_size,
                              const size_t maxThreadsPerBlock[3],
                              ur_kernel_handle_t kernel) {
  assert(threadsPerBlock != nullptr);
  assert(global_work_size != nullptr);
  assert(kernel != nullptr);
  // int recommendedBlockSize, minGrid;

  // UR_CHECK_ERROR(hipOccupancyMaxPotentialBlockSize(
  //    &minGrid, &recommendedBlockSize, kernel->get(),
  //    0, 0));

  //(void)minGrid; // Not used, avoid warnings

  threadsPerBlock[0] = std::min(maxThreadsPerBlock[0], global_work_size[0]);

  // Find a local work group size that is a divisor of the global
  // work group size to produce uniform work groups.
  while (0u != (global_work_size[0] % threadsPerBlock[0])) {
    --threadsPerBlock[0];
  }
}
} // namespace

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    size_t offset, size_t size, const void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pSrc, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_MEM_BUFFER_WRITE, hQueue, hipStream));
      retImplEv->start();
    }

    retErr = UR_CHECK_ERROR(
        hipMemcpyHtoDAsync(hBuffer->mem_.buffer_mem_.get_with_offset(offset),
                           const_cast<void *>(pSrc), size, hipStream));

    if (phEvent) {
      retErr = retImplEv->record();
    }

    if (blockingWrite) {
      retErr = UR_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (phEvent) {
      *phEvent = retImplEv.release();
    }
  } catch (ur_result_t err) {
    retErr = err;
  }
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferRead(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    size_t offset, size_t size, void *pDst, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_MEM_BUFFER_READ, hQueue, hipStream));
      retImplEv->start();
    }

    retErr = UR_CHECK_ERROR(hipMemcpyDtoHAsync(
        pDst, hBuffer->mem_.buffer_mem_.get_with_offset(offset), size,
        hipStream));

    if (phEvent) {
      retErr = retImplEv->record();
    }

    if (blockingRead) {
      retErr = UR_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (phEvent) {
      *phEvent = retImplEv.release();
    }

  } catch (ur_result_t err) {
    retErr = err;
  }
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pGlobalWorkOffset, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pGlobalWorkSize, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hQueue->get_context() == hKernel->get_context(),
            UR_RESULT_ERROR_INVALID_QUEUE);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  if (*pGlobalWorkSize == 0) {
    return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                          phEventWaitList, phEvent);
  }

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  size_t threadsPerBlock[3] = {32u, 1u, 1u};
  size_t maxWorkGroupSize = 0u;
  size_t maxThreadsPerBlock[3] = {};
  bool providedLocalWorkGroupSize = (pLocalWorkSize != nullptr);

  {
    ur_result_t retError = urDeviceGetInfo(
        hQueue->device_, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
        sizeof(maxThreadsPerBlock), maxThreadsPerBlock, nullptr);
    UR_ASSERT(retError == UR_RESULT_SUCCESS, retError);

    retError =
        urDeviceGetInfo(hQueue->device_, UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                        sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    UR_ASSERT(retError == UR_RESULT_SUCCESS, retError);

    // The maxWorkGroupsSize = 1024 for AMD GPU
    // The maxThreadsPerBlock = {1024, 1024, 1024}

    if (providedLocalWorkGroupSize) {
      auto isValid = [&](int dim) {
        UR_ASSERT(pLocalWorkSize[dim] <= maxThreadsPerBlock[dim],
                  UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
        // Checks that local work sizes are a divisor of the global work sizes
        // which includes that the local work sizes are neither larger than the
        // global work sizes and not 0.
        UR_ASSERT(pLocalWorkSize != 0, UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
        UR_ASSERT((pGlobalWorkSize[dim] % pLocalWorkSize[dim]) == 0,
                  UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);
        threadsPerBlock[dim] = pLocalWorkSize[dim];
        return UR_RESULT_SUCCESS;
      };

      for (size_t dim = 0; dim < workDim; dim++) {
        auto err = isValid(dim);
        if (err != UR_RESULT_SUCCESS)
          return err;
      }
    } else {
      simpleGuessLocalWorkSize(threadsPerBlock, pGlobalWorkSize,
                               maxThreadsPerBlock, hKernel);
    }
  }

  UR_ASSERT(maxWorkGroupSize >= size_t(threadsPerBlock[0] * threadsPerBlock[1] *
                                       threadsPerBlock[2]),
            UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);

  size_t blocksPerGrid[3] = {1u, 1u, 1u};

  for (size_t i = 0; i < workDim; i++) {
    blocksPerGrid[i] =
        (pGlobalWorkSize[i] + threadsPerBlock[i] - 1) / threadsPerBlock[i];
  }

  ur_result_t retError = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());

    uint32_t stream_token;
    ur_stream_quard guard;
    hipStream_t hipStream = hQueue->get_next_compute_stream(
        numEventsInWaitList, phEventWaitList, guard, &stream_token);
    hipFunction_t hipFunc = hKernel->get();

    retError = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                                 phEventWaitList);

    // Set the implicit global offset parameter if kernel has offset variant
    if (hKernel->get_with_offset_parameter()) {
      std::uint32_t hip_implicit_offset[3] = {0, 0, 0};
      if (pGlobalWorkOffset) {
        for (size_t i = 0; i < workDim; i++) {
          hip_implicit_offset[i] =
              static_cast<std::uint32_t>(pGlobalWorkOffset[i]);
          if (pGlobalWorkOffset[i] != 0) {
            hipFunc = hKernel->get_with_offset_parameter();
          }
        }
      }
      hKernel->set_implicit_offset_arg(sizeof(hip_implicit_offset),
                                       hip_implicit_offset);
    }

    auto argIndices = hKernel->get_arg_indices();

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_KERNEL_LAUNCH, hQueue, hipStream, stream_token));
      retImplEv->start();
    }

    // Set local mem max size if env var is present
    static const char *local_mem_sz_ptr =
        std::getenv("SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE");

    if (local_mem_sz_ptr) {
      int device_max_local_mem = 0;
      retError = UR_CHECK_ERROR(hipDeviceGetAttribute(
          &device_max_local_mem, hipDeviceAttributeMaxSharedMemoryPerBlock,
          hQueue->get_device()->get()));

      static const int env_val = std::atoi(local_mem_sz_ptr);
      if (env_val <= 0 || env_val > device_max_local_mem) {
        setErrorMessage("Invalid value specified for "
                        "SYCL_PI_HIP_MAX_LOCAL_MEM_SIZE",
                        UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      retError = UR_CHECK_ERROR(hipFuncSetAttribute(
          hipFunc, hipFuncAttributeMaxDynamicSharedMemorySize, env_val));
    }

    retError = UR_CHECK_ERROR(hipModuleLaunchKernel(
        hipFunc, blocksPerGrid[0], blocksPerGrid[1], blocksPerGrid[2],
        threadsPerBlock[0], threadsPerBlock[1], threadsPerBlock[2],
        hKernel->get_local_size(), hipStream, argIndices.data(), nullptr));

    hKernel->clear_local_size();

    if (phEvent) {
      retError = retImplEv->record();
      *phEvent = retImplEv.release();
    }
  } catch (ur_result_t err) {
    retError = err;
  }
  return retError;
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
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(!(phEventWaitList == NULL && numEventsInWaitList > 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST)
  UR_ASSERT(!(phEventWaitList != NULL && numEventsInWaitList == 0),
            UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST)

  ur_result_t result;

  try {
    ScopedContext active(hQueue->get_context());
    uint32_t stream_token;
    ur_stream_quard guard;
    hipStream_t hipStream = hQueue->get_next_compute_stream(
        numEventsInWaitList,
        reinterpret_cast<const ur_event_handle_t *>(phEventWaitList), guard,
        &stream_token);
    {
      std::lock_guard<std::mutex> guard(hQueue->barrier_mutex_);
      if (hQueue->barrier_event_ == nullptr) {
        UR_CHECK_ERROR(hipEventCreate(&hQueue->barrier_event_));
      }
      if (numEventsInWaitList == 0) { //  wait on all work
        if (hQueue->barrier_tmp_event_ == nullptr) {
          UR_CHECK_ERROR(hipEventCreate(&hQueue->barrier_tmp_event_));
        }
        hQueue->sync_streams(
            [hipStream, tmp_event = hQueue->barrier_tmp_event_](hipStream_t s) {
              if (hipStream != s) {
                UR_CHECK_ERROR(hipEventRecord(tmp_event, s));
                UR_CHECK_ERROR(hipStreamWaitEvent(hipStream, tmp_event, 0));
              }
            });
      } else { // wait just on given events
        forLatestEvents(
            reinterpret_cast<const ur_event_handle_t *>(phEventWaitList),
            numEventsInWaitList,
            [hipStream](ur_event_handle_t event) -> ur_result_t {
              if (event->get_queue()->has_been_synchronized(
                      event->get_compute_stream_token())) {
                return UR_RESULT_SUCCESS;
              } else {
                return UR_CHECK_ERROR(
                    hipStreamWaitEvent(hipStream, event->get(), 0));
              }
            });
      }

      result =
          UR_CHECK_ERROR(hipEventRecord(hQueue->barrier_event_, hipStream));
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
          UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, hQueue, hipStream, stream_token);
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

/// General 3D memory copy operation.
/// This function requires the corresponding HIP context to be at the top of
/// the context stack
/// If the source and/or destination is on the device, src_ptr and/or dst_ptr
/// must be a pointer to a hipDevPtr
static ur_result_t commonEnqueueMemBufferCopyRect(
    hipStream_t hip_stream, ur_rect_region_t region, const void *src_ptr,
    const hipMemoryType src_type, ur_rect_offset_t src_offset,
    size_t src_row_pitch, size_t src_slice_pitch, void *dst_ptr,
    const hipMemoryType dst_type, ur_rect_offset_t dst_offset,
    size_t dst_row_pitch, size_t dst_slice_pitch) {

  assert(src_type == hipMemoryTypeDevice || src_type == hipMemoryTypeHost);
  assert(dst_type == hipMemoryTypeDevice || dst_type == hipMemoryTypeHost);

  src_row_pitch = (!src_row_pitch) ? region.width : src_row_pitch;
  src_slice_pitch =
      (!src_slice_pitch) ? (region.height * src_row_pitch) : src_slice_pitch;
  dst_row_pitch = (!dst_row_pitch) ? region.width : dst_row_pitch;
  dst_slice_pitch =
      (!dst_slice_pitch) ? (region.height * dst_row_pitch) : dst_slice_pitch;

  HIP_MEMCPY3D params;

  params.WidthInBytes = region.width;
  params.Height = region.height;
  params.Depth = region.depth;

  params.srcMemoryType = src_type;
  params.srcDevice = src_type == hipMemoryTypeDevice
                         ? *static_cast<const hipDeviceptr_t *>(src_ptr)
                         : 0;
  params.srcHost = src_type == hipMemoryTypeHost ? src_ptr : nullptr;
  params.srcXInBytes = src_offset.x;
  params.srcY = src_offset.y;
  params.srcZ = src_offset.z;
  params.srcPitch = src_row_pitch;
  params.srcHeight = src_slice_pitch / src_row_pitch;

  params.dstMemoryType = dst_type;
  params.dstDevice = dst_type == hipMemoryTypeDevice
                         ? *reinterpret_cast<hipDeviceptr_t *>(dst_ptr)
                         : 0;
  params.dstHost = dst_type == hipMemoryTypeHost ? dst_ptr : nullptr;
  params.dstXInBytes = dst_offset.x;
  params.dstY = dst_offset.y;
  params.dstZ = dst_offset.z;
  params.dstPitch = dst_row_pitch;
  params.dstHeight = dst_slice_pitch / dst_row_pitch;

  return UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&params, hip_stream));
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferReadRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingRead,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
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

  ur_result_t retErr = UR_RESULT_SUCCESS;
  void *devPtr = hBuffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();

    retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_MEM_BUFFER_READ_RECT, hQueue, hipStream));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, &devPtr, hipMemoryTypeDevice, bufferOrigin,
        bufferRowPitch, bufferSlicePitch, pDst, hipMemoryTypeHost, hostOrigin,
        hostRowPitch, hostSlicePitch);

    if (phEvent) {
      retErr = retImplEv->record();
    }

    if (blockingRead) {
      retErr = UR_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (phEvent) {
      *phEvent = retImplEv.release();
    }

  } catch (ur_result_t err) {
    retErr = err;
  }
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferWriteRect(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, bool blockingWrite,
    ur_rect_offset_t bufferOrigin, ur_rect_offset_t hostOrigin,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  void *devPtr = hBuffer->mem_.buffer_mem_.get_void();
  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_MEM_BUFFER_WRITE_RECT, hQueue, hipStream));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, pSrc, hipMemoryTypeHost, hostOrigin, hostRowPitch,
        hostSlicePitch, &devPtr, hipMemoryTypeDevice, bufferOrigin,
        bufferRowPitch, bufferSlicePitch);

    if (phEvent) {
      retErr = retImplEv->record();
    }

    if (blockingWrite) {
      retErr = UR_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }

    if (phEvent) {
      *phEvent = retImplEv.release();
    }

  } catch (ur_result_t err) {
    retErr = err;
  }
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBufferSrc,
    ur_mem_handle_t hBufferDst, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    ur_result_t result;
    auto stream = hQueue->get_next_transfer_stream();

    if (phEventWaitList) {
      result = enqueueEventsWait(hQueue, stream, numEventsInWaitList,
                                 phEventWaitList);
    }

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_MEM_BUFFER_COPY, hQueue, stream));
      result = retImplEv->start();
    }

    auto src = hBufferSrc->mem_.buffer_mem_.get_with_offset(srcOffset);
    auto dst = hBufferDst->mem_.buffer_mem_.get_with_offset(dstOffset);

    result = UR_CHECK_ERROR(hipMemcpyDtoDAsync(dst, src, size, stream));

    if (phEvent) {
      result = retImplEv->record();
      *phEvent = retImplEv.release();
    }

    return result;
  } catch (ur_result_t err) {
    return err;
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

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBufferSrc, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBufferDst, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  void *srcPtr = hBufferSrc->mem_.buffer_mem_.get_void();
  void *dstPtr = hBufferDst->mem_.buffer_mem_.get_void();
  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_MEM_BUFFER_COPY_RECT, hQueue, hipStream));
      retImplEv->start();
    }

    retErr = commonEnqueueMemBufferCopyRect(
        hipStream, region, &srcPtr, hipMemoryTypeDevice, srcOrigin, srcRowPitch,
        srcSlicePitch, &dstPtr, hipMemoryTypeDevice, dstOrigin, dstRowPitch,
        dstSlicePitch);

    if (phEvent) {
      retImplEv->record();
      *phEvent = retImplEv.release();
    }

  } catch (ur_result_t err) {
    retErr = err;
  }
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemBufferFill(
    ur_queue_handle_t hQueue, ur_mem_handle_t hBuffer, const void *pPattern,
    size_t patternSize, size_t offset, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pPattern, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto args_are_multiples_of_pattern_size =
      (offset % patternSize == 0) || (size % patternSize == 0);

  auto pattern_is_valid = (pPattern != nullptr);

  auto pattern_size_is_valid =
      ((patternSize & (patternSize - 1)) == 0) && // is power of two
      (patternSize > 0) && (patternSize <= 128);  // falls within valid range

  UR_ASSERT(args_are_multiples_of_pattern_size && pattern_is_valid &&
                pattern_size_is_valid,
            UR_RESULT_ERROR_INVALID_VALUE);
  (void)args_are_multiples_of_pattern_size;
  (void)pattern_is_valid;
  (void)pattern_size_is_valid;

  std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

  try {
    ScopedContext active(hQueue->get_context());

    auto stream = hQueue->get_next_transfer_stream();
    ur_result_t result;
    if (phEventWaitList) {
      result = enqueueEventsWait(hQueue, stream, numEventsInWaitList,
                                 phEventWaitList);
    }

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_MEM_BUFFER_FILL, hQueue, stream));
      result = retImplEv->start();
    }

    auto dstDevice = hBuffer->mem_.buffer_mem_.get_with_offset(offset);
    auto N = size / patternSize;

    // pattern size in bytes
    switch (patternSize) {
    case 1: {
      auto value = *static_cast<const uint8_t *>(pPattern);
      result = UR_CHECK_ERROR(hipMemsetD8Async(dstDevice, value, N, stream));
      break;
    }
    case 2: {
      auto value = *static_cast<const uint16_t *>(pPattern);
      result = UR_CHECK_ERROR(hipMemsetD16Async(dstDevice, value, N, stream));
      break;
    }
    case 4: {
      auto value = *static_cast<const uint32_t *>(pPattern);
      result = UR_CHECK_ERROR(hipMemsetD32Async(dstDevice, value, N, stream));
      break;
    }

    default: {
      // HIP has no memset functions that allow setting values more than 4
      // bytes. UR API lets you pass an arbitrary "pattern" to the buffer
      // fill, which can be more than 4 bytes. We must break up the pattern
      // into 1 byte values, and set the buffer using multiple strided calls.
      // The first 4 patterns are set using hipMemsetD32Async then all
      // subsequent 1 byte patterns are set using hipMemset2DAsync which is
      // called for each pattern.

      // Calculate the number of patterns, stride, number of times the pattern
      // needs to be applied, and the number of times the first 32 bit pattern
      // needs to be applied.
      auto number_of_steps = patternSize / sizeof(uint8_t);
      auto pitch = number_of_steps * sizeof(uint8_t);
      auto height = size / number_of_steps;
      auto count_32 = size / sizeof(uint32_t);

      // Get 4-byte chunk of the pattern and call hipMemsetD32Async
      auto value = *(static_cast<const uint32_t *>(pPattern));
      result =
          UR_CHECK_ERROR(hipMemsetD32Async(dstDevice, value, count_32, stream));
      for (auto step = 4u; step < number_of_steps; ++step) {
        // take 1 byte of the pattern
        value = *(static_cast<const uint8_t *>(pPattern) + step);

        // offset the pointer to the part of the buffer we want to write to
        auto offset_ptr = reinterpret_cast<void *>(
            reinterpret_cast<uint8_t *>(dstDevice) + (step * sizeof(uint8_t)));

        // set all of the pattern chunks
        result = UR_CHECK_ERROR(hipMemset2DAsync(
            offset_ptr, pitch, value, sizeof(uint8_t), height, stream));
      }
      break;
    }
    }

    if (phEvent) {
      result = retImplEv->record();
      *phEvent = retImplEv.release();
    }

    return result;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

/// General ND memory copy operation for images (where N > 1).
/// This function requires the corresponding HIP context to be at the top of
/// the context stack
/// If the source and/or destination is an array, src_ptr and/or dst_ptr
/// must be a pointer to a hipArray
static ur_result_t commonEnqueueMemImageNDCopy(
    hipStream_t hip_stream, ur_mem_type_t img_type, const size_t *region,
    const void *src_ptr, const hipMemoryType src_type, const size_t *src_offset,
    void *dst_ptr, const hipMemoryType dst_type, const size_t *dst_offset) {
  UR_ASSERT(region, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT(src_type == hipMemoryTypeArray || src_type == hipMemoryTypeHost,
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(dst_type == hipMemoryTypeArray || dst_type == hipMemoryTypeHost,
            UR_RESULT_ERROR_INVALID_VALUE);

  if (img_type == UR_MEM_TYPE_IMAGE2D) {
    hip_Memcpy2D cpyDesc;
    memset(&cpyDesc, 0, sizeof(cpyDesc));
    cpyDesc.srcMemoryType = src_type;
    if (src_type == hipMemoryTypeArray) {
      cpyDesc.srcArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(src_ptr));
      cpyDesc.srcXInBytes = src_offset[0];
      cpyDesc.srcY = src_offset[1];
    } else {
      cpyDesc.srcHost = src_ptr;
    }
    cpyDesc.dstMemoryType = dst_type;
    if (dst_type == hipMemoryTypeArray) {
      cpyDesc.dstArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(dst_ptr));
      cpyDesc.dstXInBytes = dst_offset[0];
      cpyDesc.dstY = dst_offset[1];
    } else {
      cpyDesc.dstHost = dst_ptr;
    }
    cpyDesc.WidthInBytes = region[0];
    cpyDesc.Height = region[1];
    return UR_CHECK_ERROR(hipMemcpyParam2DAsync(&cpyDesc, hip_stream));
  }

  if (img_type == UR_MEM_TYPE_IMAGE3D) {

    HIP_MEMCPY3D cpyDesc;
    memset(&cpyDesc, 0, sizeof(cpyDesc));
    cpyDesc.srcMemoryType = src_type;
    if (src_type == hipMemoryTypeArray) {
      cpyDesc.srcArray =
          reinterpret_cast<hipCUarray>(const_cast<void *>(src_ptr));
      cpyDesc.srcXInBytes = src_offset[0];
      cpyDesc.srcY = src_offset[1];
      cpyDesc.srcZ = src_offset[2];
    } else {
      cpyDesc.srcHost = src_ptr;
    }
    cpyDesc.dstMemoryType = dst_type;
    if (dst_type == hipMemoryTypeArray) {
      cpyDesc.dstArray = reinterpret_cast<hipCUarray>(dst_ptr);
      cpyDesc.dstXInBytes = dst_offset[0];
      cpyDesc.dstY = dst_offset[1];
      cpyDesc.dstZ = dst_offset[2];
    } else {
      cpyDesc.dstHost = dst_ptr;
    }
    cpyDesc.WidthInBytes = region[0];
    cpyDesc.Height = region[1];
    cpyDesc.Depth = region[2];
    return UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpyDesc, hip_stream));
    return UR_RESULT_ERROR_UNKNOWN;
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
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hImage, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hImage->mem_type_ == ur_mem_handle_t_::mem_type::surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t retErr = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();

    if (phEventWaitList) {
      retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                                 phEventWaitList);
    }

    hipArray *array = hImage->mem_.surface_mem_.get_array();

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(array, Format, NumChannels);

    int elementByteSize = imageElementByteSize(Format);

    size_t byteOffsetX = origin.x * elementByteSize * NumChannels;
    size_t bytesToCopy = elementByteSize * NumChannels * region.depth;

    auto imgType = hImage->mem_.surface_mem_.get_image_type();

    size_t adjustedRegion[3] = {bytesToCopy, region.height, region.height};
    size_t srcOffset[3] = {byteOffsetX, origin.y, origin.z};

    retErr = commonEnqueueMemImageNDCopy(hipStream, imgType, adjustedRegion,
                                         array, hipMemoryTypeArray, srcOffset,
                                         pDst, hipMemoryTypeHost, nullptr);

    if (retErr != UR_RESULT_SUCCESS) {
      return retErr;
    }

    if (phEvent) {
      auto new_event = ur_event_handle_t_::make_native(
          UR_COMMAND_MEM_IMAGE_READ, hQueue, hipStream);
      new_event->record();
      *phEvent = new_event;
    }

    if (blockingRead) {
      retErr = UR_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageWrite(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImage, bool blockingWrite,
    ur_rect_offset_t origin, ur_rect_region_t region, size_t rowPitch,
    size_t slicePitch, void *pSrc, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  std::ignore = blockingWrite;
  std::ignore = rowPitch;
  std::ignore = slicePitch;
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hImage, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pSrc, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hImage->mem_type_ == ur_mem_handle_t_::mem_type::surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t retErr = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();

    if (phEventWaitList) {
      retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                                 phEventWaitList);
    }

    hipArray *array = hImage->mem_.surface_mem_.get_array();

    hipArray_Format Format;
    size_t NumChannels;
    getArrayDesc(array, Format, NumChannels);

    int elementByteSize = imageElementByteSize(Format);

    size_t byteOffsetX = origin.x * elementByteSize * NumChannels;
    size_t bytesToCopy = elementByteSize * NumChannels * region.depth;

    auto imgType = hImage->mem_.surface_mem_.get_image_type();

    size_t adjustedRegion[3] = {bytesToCopy, region.height, region.height};
    size_t dstOffset[3] = {byteOffsetX, origin.y, origin.z};

    retErr = commonEnqueueMemImageNDCopy(hipStream, imgType, adjustedRegion,
                                         pSrc, hipMemoryTypeHost, nullptr,
                                         array, hipMemoryTypeArray, dstOffset);

    if (retErr != UR_RESULT_SUCCESS) {
      return retErr;
    }

    if (phEvent) {
      auto new_event = ur_event_handle_t_::make_native(
          UR_COMMAND_MEM_IMAGE_WRITE, hQueue, hipStream);
      new_event->record();
      *phEvent = new_event;
    }
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;

  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemImageCopy(
    ur_queue_handle_t hQueue, ur_mem_handle_t hImageSrc,
    ur_mem_handle_t hImageDst, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE)
  UR_ASSERT(hImageSrc, UR_RESULT_ERROR_INVALID_NULL_HANDLE)
  UR_ASSERT(hImageDst, UR_RESULT_ERROR_INVALID_NULL_HANDLE)
  UR_ASSERT(hImageSrc->mem_type_ == ur_mem_handle_t_::mem_type::surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hImageDst->mem_type_ == ur_mem_handle_t_::mem_type::surface,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hImageSrc->mem_.surface_mem_.get_image_type() ==
                hImageDst->mem_.surface_mem_.get_image_type(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t retErr = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    if (phEventWaitList) {
      retErr = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                                 phEventWaitList);
    }

    hipArray *srcArray = hImageSrc->mem_.surface_mem_.get_array();
    hipArray_Format srcFormat;
    size_t srcNumChannels;
    getArrayDesc(srcArray, srcFormat, srcNumChannels);

    hipArray *dstArray = hImageDst->mem_.surface_mem_.get_array();
    hipArray_Format dstFormat;
    size_t dstNumChannels;
    getArrayDesc(dstArray, dstFormat, dstNumChannels);

    UR_ASSERT(srcFormat == dstFormat,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    UR_ASSERT(srcNumChannels == dstNumChannels,
              UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR);

    int elementByteSize = imageElementByteSize(srcFormat);

    size_t dstByteOffsetX = dstOrigin.x * elementByteSize * srcNumChannels;
    size_t srcByteOffsetX = srcOrigin.x * elementByteSize * dstNumChannels;
    size_t bytesToCopy = elementByteSize * srcNumChannels * region.depth;

    auto imgType = hImageSrc->mem_.surface_mem_.get_image_type();

    size_t adjustedRegion[3] = {bytesToCopy, region.height, region.width};
    size_t srcOffset[3] = {srcByteOffsetX, srcOrigin.y, srcOrigin.z};
    size_t dstOffset[3] = {dstByteOffsetX, dstOrigin.y, dstOrigin.z};

    retErr = commonEnqueueMemImageNDCopy(
        hipStream, imgType, adjustedRegion, srcArray, hipMemoryTypeArray,
        srcOffset, dstArray, hipMemoryTypeArray, dstOffset);

    if (retErr != UR_RESULT_SUCCESS) {
      return retErr;
    }

    if (phEvent) {
      auto new_event = ur_event_handle_t_::make_native(
          UR_COMMAND_MEM_IMAGE_COPY, hQueue, hipStream);
      new_event->record();
      *phEvent = new_event;
    }
  } catch (ur_result_t err) {
    return err;
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
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(ppRetMap, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hBuffer->mem_type_ == ur_mem_handle_t_::mem_type::buffer,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  ur_result_t ret_err = UR_RESULT_ERROR_INVALID_OPERATION;
  const bool is_pinned =
      hBuffer->mem_.buffer_mem_.allocMode_ ==
      ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::alloc_host_ptr;

  // Currently no support for overlapping regions
  if (hBuffer->mem_.buffer_mem_.get_map_ptr() != nullptr) {
    return ret_err;
  }

  // Allocate a pointer in the host to store the mapped information
  auto hostPtr = hBuffer->mem_.buffer_mem_.map_to_ptr(offset, mapFlags);
  *ppRetMap = hBuffer->mem_.buffer_mem_.get_map_ptr();
  if (hostPtr) {
    ret_err = UR_RESULT_SUCCESS;
  }

  if (!is_pinned &&
      ((mapFlags & UR_MAP_FLAG_READ) || (mapFlags & UR_MAP_FLAG_WRITE))) {
    // Pinned host memory is already on host so it doesn't need to be read.
    ret_err = urEnqueueMemBufferRead(hQueue, hBuffer, blockingMap, offset, size,
                                     hostPtr, numEventsInWaitList,
                                     phEventWaitList, phEvent);
  } else {
    ScopedContext active(hQueue->get_context());

    if (is_pinned) {
      ret_err = urEnqueueEventsWait(hQueue, numEventsInWaitList,
                                    phEventWaitList, nullptr);
    }

    if (phEvent) {
      try {
        *phEvent =
            ur_event_handle_t_::make_native(UR_COMMAND_MEM_BUFFER_MAP, hQueue,
                                            hQueue->get_next_transfer_stream());
        (*phEvent)->start();
        (*phEvent)->record();
      } catch (ur_result_t error) {
        ret_err = error;
      }
    }
  }

  return ret_err;
}

/// Implements the unmap from the host, using a BufferWrite operation.
/// Requires the mapped pointer to be already registered in the given hMem.
/// If hMem uses pinned host memory, this will not do a write.
///
UR_APIEXPORT ur_result_t UR_APICALL urEnqueueMemUnmap(
    ur_queue_handle_t hQueue, ur_mem_handle_t hMem, void *pMappedPtr,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t ret_err = UR_RESULT_SUCCESS;

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pMappedPtr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hMem->mem_type_ == ur_mem_handle_t_::mem_type::buffer,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hMem->mem_.buffer_mem_.get_map_ptr() != nullptr,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);
  UR_ASSERT(hMem->mem_.buffer_mem_.get_map_ptr() == pMappedPtr,
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  const bool is_pinned =
      hMem->mem_.buffer_mem_.allocMode_ ==
      ur_mem_handle_t_::mem_::buffer_mem_::alloc_mode::alloc_host_ptr;

  if (!is_pinned &&
      ((hMem->mem_.buffer_mem_.get_map_flags() & UR_MAP_FLAG_WRITE) ||
       (hMem->mem_.buffer_mem_.get_map_flags() &
        UR_MAP_FLAG_WRITE_INVALIDATE_REGION))) {
    // Pinned host memory is only on host so it doesn't need to be written to.
    ret_err = urEnqueueMemBufferWrite(
        hQueue, hMem, true, hMem->mem_.buffer_mem_.get_map_offset(pMappedPtr),
        hMem->mem_.buffer_mem_.get_size(), pMappedPtr, numEventsInWaitList,
        phEventWaitList, phEvent);
  } else {
    ScopedContext active(hQueue->get_context());

    if (is_pinned) {
      ret_err = urEnqueueEventsWait(hQueue, numEventsInWaitList,
                                    phEventWaitList, nullptr);
    }

    if (phEvent) {
      try {
        *phEvent = ur_event_handle_t_::make_native(
            UR_COMMAND_MEM_UNMAP, hQueue, hQueue->get_next_transfer_stream());
        (*phEvent)->start();
        (*phEvent)->record();
      } catch (ur_result_t error) {
        ret_err = error;
      }
    }
  }

  hMem->mem_.buffer_mem_.unmap(pMappedPtr);
  return ret_err;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill(
    ur_queue_handle_t hQueue, void *ptr, size_t patternSize,
    const void *pPattern, size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(ptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pPattern, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> event_ptr{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    uint32_t stream_token;
    ur_stream_quard guard;
    hipStream_t hipStream = hQueue->get_next_compute_stream(
        numEventsInWaitList, phEventWaitList, guard, &stream_token);
    result = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);
    if (phEvent) {
      event_ptr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_USM_FILL, hQueue, hipStream, stream_token));
      event_ptr->start();
    }
    switch (patternSize) {
    case 1:
      result = UR_CHECK_ERROR(
          hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(ptr),
                           *(const uint8_t *)pPattern & 0xFF, size, hipStream));
      break;
    case 2:
      result = UR_CHECK_ERROR(hipMemsetD16Async(
          reinterpret_cast<hipDeviceptr_t>(ptr),
          *(const uint16_t *)pPattern & 0xFFFF, size, hipStream));
      break;
    case 4:
      result = UR_CHECK_ERROR(hipMemsetD32Async(
          reinterpret_cast<hipDeviceptr_t>(ptr),
          *(const uint32_t *)pPattern & 0xFFFFFFFF, size, hipStream));
      break;

    default:
      return UR_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (phEvent) {
      result = event_ptr->record();
      *phEvent = event_ptr.release();
    }
  } catch (ur_result_t err) {
    result = err;
  }

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy(
    ur_queue_handle_t hQueue, bool blocking, void *pDst, const void *pSrc,
    size_t size, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pSrc, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t result = UR_RESULT_SUCCESS;

  std::unique_ptr<ur_event_handle_t_> event_ptr{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    result = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);
    if (phEvent) {
      event_ptr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_USM_MEMCPY, hQueue, hipStream));
      event_ptr->start();
    }
    result = UR_CHECK_ERROR(
        hipMemcpyAsync(pDst, pSrc, size, hipMemcpyDefault, hipStream));
    if (phEvent) {
      result = event_ptr->record();
    }
    if (blocking) {
      result = UR_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }
    if (phEvent) {
      *phEvent = event_ptr.release();
    }
  } catch (ur_result_t err) {
    result = err;
  }
  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMPrefetch(
    ur_queue_handle_t hQueue, const void *pMem, size_t size,
    ur_usm_migration_flags_t flags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  // flags is currently unused so fail if set
  if (flags != 0)
    return UR_RESULT_ERROR_INVALID_VALUE;
  ur_result_t result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_event_handle_t_> event_ptr{nullptr};

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    result = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);
    if (phEvent) {
      event_ptr =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_USM_PREFETCH, hQueue, hipStream));
      event_ptr->start();
    }
    result = UR_CHECK_ERROR(hipMemPrefetchAsync(
        pMem, size, hQueue->get_context()->get_device()->get(), hipStream));
    if (phEvent) {
      result = event_ptr->record();
      *phEvent = event_ptr.release();
    }
  } catch (ur_result_t err) {
    result = err;
  }

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urEnqueueUSMAdvise(ur_queue_handle_t hQueue, const void *pMem, size_t size,
                   ur_usm_advice_flags_t advice, ur_event_handle_t *phEvent) {
  std::ignore = size;
  std::ignore = advice;

  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  // TODO implement a mapping to hipMemAdvise once the expected behaviour
  // of urEnqueueUSMAdvise is detailed in the USM extension
  return urEnqueueEventsWait(hQueue, 0, nullptr, phEvent);
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t hQueue, void *pMem, size_t pitch, size_t patternSize,
    const void *pPattern, size_t width, size_t height,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  std::ignore = hQueue;
  std::ignore = pMem;
  std::ignore = pitch;
  std::ignore = patternSize;
  std::ignore = pPattern;
  std::ignore = width;
  std::ignore = height;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

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
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pDst, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pSrc, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t result = UR_RESULT_SUCCESS;

  try {
    ScopedContext active(hQueue->get_context());
    hipStream_t hipStream = hQueue->get_next_transfer_stream();
    result = enqueueEventsWait(hQueue, hipStream, numEventsInWaitList,
                               phEventWaitList);
    if (phEvent) {
      (*phEvent) = ur_event_handle_t_::make_native(UR_COMMAND_USM_MEMCPY_2D,
                                                   hQueue, hipStream);
      (*phEvent)->start();
    }

    result =
        UR_CHECK_ERROR(hipMemcpy2DAsync(pDst, dstPitch, pSrc, srcPitch, width,
                                        height, hipMemcpyDefault, hipStream));

    if (phEvent) {
      (*phEvent)->record();
    }
    if (blocking) {
      result = UR_CHECK_ERROR(hipStreamSynchronize(hipStream));
    }
  } catch (ur_result_t err) {
    result = err;
  }

  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableWrite(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingWrite, size_t count, size_t offset, const void *pSrc,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = name;
  std::ignore = blockingWrite;
  std::ignore = count;
  std::ignore = offset;
  std::ignore = pSrc;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueDeviceGlobalVariableRead(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram, const char *name,
    bool blockingRead, size_t count, size_t offset, void *pDst,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = name;
  std::ignore = blockingRead;
  std::ignore = count;
  std::ignore = offset;
  std::ignore = pDst;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueReadHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pDst, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pDst;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueWriteHostPipe(
    ur_queue_handle_t hQueue, ur_program_handle_t hProgram,
    const char *pipe_symbol, bool blocking, void *pSrc, size_t size,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  std::ignore = hQueue;
  std::ignore = hProgram;
  std::ignore = pipe_symbol;
  std::ignore = blocking;
  std::ignore = pSrc;
  std::ignore = size;
  std::ignore = numEventsInWaitList;
  std::ignore = phEventWaitList;
  std::ignore = phEvent;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
