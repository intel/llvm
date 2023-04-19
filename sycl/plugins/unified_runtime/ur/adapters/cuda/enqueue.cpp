//===--------- enqueue.cpp - CUDA Adapter ----------------------------===//
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
#include "queue.hpp"

#include <cmath>
#include <cuda.h>

ur_result_t enqueueEventsWait(ur_queue_handle_t command_queue, CUstream stream,
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
            return UR_CHECK_ERROR(cuStreamWaitEvent(stream, event->get(), 0));
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

// Determine local work sizes that result in uniform work groups.
// The default threadsPerBlock only require handling the first work_dim
// dimension.
void guessLocalWorkSize(ur_device_handle_t device, size_t *threadsPerBlock,
                        const size_t *global_work_size,
                        const size_t maxThreadsPerBlock[3],
                        ur_kernel_handle_t kernel, uint32_t local_size) {
  assert(threadsPerBlock != nullptr);
  assert(global_work_size != nullptr);
  assert(kernel != nullptr);
  int minGrid, maxBlockSize, maxBlockDim[3];

  static auto isPrime = [](size_t number) -> bool {
    auto lastNumToCheck = ceil(sqrt(number));
    if (number < 2)
      return false;
    if (number == 2)
      return true;
    if (number % 2 == 0)
      return false;
    for (int i = 3; i <= lastNumToCheck; i += 2) {
      if (number % i == 0)
        return false;
    }
    return true;
  };

  cuDeviceGetAttribute(&maxBlockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                       device->get());
  cuDeviceGetAttribute(&maxBlockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                       device->get());

  UR_CHECK_ERROR(cuOccupancyMaxPotentialBlockSize(
      &minGrid, &maxBlockSize, kernel->get(), NULL, local_size,
      maxThreadsPerBlock[0]));

  threadsPerBlock[2] = std::min(global_work_size[2], size_t(maxBlockDim[2]));
  threadsPerBlock[1] =
      std::min(global_work_size[1], std::min(maxBlockSize / threadsPerBlock[2],
                                             size_t(maxBlockDim[1])));
  maxBlockDim[0] = maxBlockSize / (threadsPerBlock[1] * threadsPerBlock[2]);
  threadsPerBlock[0] =
      std::min(maxThreadsPerBlock[0],
               std::min(global_work_size[0], size_t(maxBlockDim[0])));

  // When global_work_size[0] is prime threadPerBlock[0] will later computed as
  // 1, which is not efficient configuration. In such case we use
  // global_work_size[0] + 1 to compute threadPerBlock[0].
  int adjusted_0_dim_global_work_size =
      (isPrime(global_work_size[0]) &&
       (threadsPerBlock[0] != global_work_size[0]))
          ? global_work_size[0] + 1
          : global_work_size[0];

  static auto isPowerOf2 = [](size_t value) -> bool {
    return value && !(value & (value - 1));
  };

  // Find a local work group size that is a divisor of the global
  // work group size to produce uniform work groups.
  // Additionally, for best compute utilisation, the local size has
  // to be a power of two.
  while (0u != (adjusted_0_dim_global_work_size % threadsPerBlock[0]) ||
         !isPowerOf2(threadsPerBlock[0])) {
    --threadsPerBlock[0];
  }
}

// Helper to verify out-of-registers case (exceeded block max registers).
// If the kernel requires a number of registers for the entire thread
// block exceeds the hardware limitations, then the cuLaunchKernel call
// will fail to launch with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES error.
bool hasExceededMaxRegistersPerBlock(ur_device_handle_t device,
                                     ur_kernel_handle_t kernel,
                                     size_t blockSize) {
  int maxRegsPerBlock{0};
  UR_CHECK_ERROR(cuDeviceGetAttribute(
      &maxRegsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      device->get()));

  int regsPerThread{0};
  UR_CHECK_ERROR(cuFuncGetAttribute(&regsPerThread, CU_FUNC_ATTRIBUTE_NUM_REGS,
                                    kernel->get()));

  return blockSize * regsPerThread > size_t(maxRegsPerBlock);
};

/// Enqueues a wait on the given CUstream for all specified events (See
/// \ref enqueueEventWaitWithBarrier.) If the events list is empty, the enqueued
/// wait will wait on all previous events in the queue.
///
UR_DLLEXPORT ur_result_t UR_APICALL urEnqueueEventsWaitWithBarrier(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // This function makes one stream work on the previous work (or work
  // represented by input events) and then all future work waits on that stream.
  if (!hQueue) {
    return UR_RESULT_ERROR_INVALID_QUEUE;
  }

  ur_result_t result;

  try {
    ScopedContext active(hQueue->get_context());
    uint32_t stream_token;
    ur_stream_guard_ guard;
    CUstream cuStream = hQueue->get_next_compute_stream(
        numEventsInWaitList, phEventWaitList, guard, &stream_token);
    {
      std::lock_guard<std::mutex> guard(hQueue->barrier_mutex_);
      if (hQueue->barrier_event_ == nullptr) {
        UR_CHECK_ERROR(
            cuEventCreate(&hQueue->barrier_event_, CU_EVENT_DISABLE_TIMING));
      }
      if (numEventsInWaitList == 0) { //  wait on all work
        if (hQueue->barrier_tmp_event_ == nullptr) {
          UR_CHECK_ERROR(cuEventCreate(&hQueue->barrier_tmp_event_,
                                       CU_EVENT_DISABLE_TIMING));
        }
        hQueue->sync_streams(
            [cuStream, tmp_event = hQueue->barrier_tmp_event_](CUstream s) {
              if (cuStream != s) {
                // record a new CUDA event on every stream and make one stream
                // wait for these events
                UR_CHECK_ERROR(cuEventRecord(tmp_event, s));
                UR_CHECK_ERROR(cuStreamWaitEvent(cuStream, tmp_event, 0));
              }
            });
      } else { // wait just on given events
        forLatestEvents(phEventWaitList, numEventsInWaitList,
                        [cuStream](ur_event_handle_t event) -> ur_result_t {
                          if (event->get_queue()->has_been_synchronized(
                                  event->get_compute_stream_token())) {
                            return UR_RESULT_SUCCESS;
                          } else {
                            return UR_CHECK_ERROR(
                                cuStreamWaitEvent(cuStream, event->get(), 0));
                          }
                        });
      }

      result = UR_CHECK_ERROR(cuEventRecord(hQueue->barrier_event_, cuStream));
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
          UR_COMMAND_EVENTS_WAIT_WITH_BARRIER, hQueue, cuStream, stream_token);
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

/// Enqueues a wait on the given CUstream for all events.
/// See \ref enqueueEventWait
/// TODO: Add support for multiple streams once the Event class is properly
/// refactored.
///
UR_DLLEXPORT ur_result_t UR_APICALL urEnqueueEventsWait(
    ur_queue_handle_t hQueue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  return urEnqueueEventsWaitWithBarrier(hQueue, numEventsInWaitList,
                                        phEventWaitList, phEvent);
}

UR_DLLEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  // Preconditions
  UR_ASSERT(hQueue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hQueue->get_context() == hKernel->get_context(),
            UR_RESULT_ERROR_INVALID_KERNEL);
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pGlobalWorkOffset, UR_RESULT_ERROR_INVALID_NULL_POINTER);
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
  int32_t local_size = hKernel->get_local_size();
  ur_result_t retError = UR_RESULT_SUCCESS;

  try {
    // Set the active context here as guessLocalWorkSize needs an active context
    ScopedContext active(hQueue->get_context());
    {
      size_t *reqdThreadsPerBlock = hKernel->reqdThreadsPerBlock_;
      maxWorkGroupSize = hQueue->device_->get_max_work_group_size();
      hQueue->device_->get_max_work_item_sizes(sizeof(maxThreadsPerBlock),
                                               maxThreadsPerBlock);

      if (providedLocalWorkGroupSize) {
        auto isValid = [&](int dim) {
          if (reqdThreadsPerBlock[dim] != 0 &&
              pLocalWorkSize[dim] != reqdThreadsPerBlock[dim])
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;

          if (pLocalWorkSize[dim] > maxThreadsPerBlock[dim])
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
          // Checks that local work sizes are a divisor of the global work sizes
          // which includes that the local work sizes are neither larger than
          // the global work sizes and not 0.
          if (0u == pLocalWorkSize[dim])
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
          if (0u != (pGlobalWorkSize[dim] % pLocalWorkSize[dim]))
            return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
          threadsPerBlock[dim] = pLocalWorkSize[dim];
          return UR_RESULT_SUCCESS;
        };

        size_t kernelLocalWorkGroupSize = 0;
        for (size_t dim = 0; dim < workDim; dim++) {
          auto err = isValid(dim);
          if (err != UR_RESULT_SUCCESS)
            return err;
          // If no error then sum the total local work size per dim.
          kernelLocalWorkGroupSize += pLocalWorkSize[dim];
        }

        if (hasExceededMaxRegistersPerBlock(hQueue->device_, hKernel,
                                            kernelLocalWorkGroupSize)) {
          return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
        }
      } else {
        guessLocalWorkSize(hQueue->device_, threadsPerBlock, pGlobalWorkSize,
                           maxThreadsPerBlock, hKernel, local_size);
      }
    }

    if (maxWorkGroupSize <
        size_t(threadsPerBlock[0] * threadsPerBlock[1] * threadsPerBlock[2])) {
      return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
    }

    size_t blocksPerGrid[3] = {1u, 1u, 1u};

    for (size_t i = 0; i < workDim; i++) {
      blocksPerGrid[i] =
          (pGlobalWorkSize[i] + threadsPerBlock[i] - 1) / threadsPerBlock[i];
    }

    std::unique_ptr<ur_event_handle_t_> retImplEv{nullptr};

    uint32_t stream_token;
    ur_stream_guard_ guard;
    CUstream cuStream = hQueue->get_next_compute_stream(
        numEventsInWaitList, phEventWaitList, guard, &stream_token);
    CUfunction cuFunc = hKernel->get();

    retError = enqueueEventsWait(hQueue, cuStream, numEventsInWaitList,
                                 phEventWaitList);

    // Set the implicit global offset parameter if kernel has offset variant
    if (hKernel->get_with_offset_parameter()) {
      std::uint32_t cuda_implicit_offset[3] = {0, 0, 0};
      if (pGlobalWorkOffset) {
        for (size_t i = 0; i < workDim; i++) {
          cuda_implicit_offset[i] =
              static_cast<std::uint32_t>(pGlobalWorkOffset[i]);
          if (pGlobalWorkOffset[i] != 0) {
            cuFunc = hKernel->get_with_offset_parameter();
          }
        }
      }
      hKernel->set_implicit_offset_arg(sizeof(cuda_implicit_offset),
                                       cuda_implicit_offset);
    }

    auto &argIndices = hKernel->get_arg_indices();

    if (phEvent) {
      retImplEv =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::make_native(
              UR_COMMAND_KERNEL_LAUNCH, hQueue, cuStream, stream_token));
      retImplEv->start();
    }

    // Set local mem max size if env var is present
    static const char *local_mem_sz_ptr =
        std::getenv("SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE");

    if (local_mem_sz_ptr) {
      int device_max_local_mem = 0;
      cuDeviceGetAttribute(
          &device_max_local_mem,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
          hQueue->get_device()->get());

      static const int env_val = std::atoi(local_mem_sz_ptr);
      if (env_val <= 0 || env_val > device_max_local_mem) {
        setErrorMessage("Invalid value specified for "
                        "SYCL_PI_CUDA_MAX_LOCAL_MEM_SIZE",
                        UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      UR_CHECK_ERROR(cuFuncSetAttribute(
          cuFunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, env_val));
    }

    retError = UR_CHECK_ERROR(cuLaunchKernel(
        cuFunc, blocksPerGrid[0], blocksPerGrid[1], blocksPerGrid[2],
        threadsPerBlock[0], threadsPerBlock[1], threadsPerBlock[2], local_size,
        cuStream, const_cast<void **>(argIndices.data()), nullptr));
    if (local_size != 0)
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
