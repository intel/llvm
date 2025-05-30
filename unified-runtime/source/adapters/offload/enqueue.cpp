//===----------- enqueue.cpp - LLVM Offload Adapter  ----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <assert.h>
#include <ur_api.h>

#include "event.hpp"
#include "kernel.hpp"
#include "queue.hpp"
#include "ur2offload.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunch(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    const size_t *pLocalWorkSize, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  // Ignore wait list for now
  (void)numEventsInWaitList;
  (void)phEventWaitList;
  //

  (void)pGlobalWorkOffset;
  (void)pLocalWorkSize;

  if (workDim == 1) {
    std::cerr
        << "UR Offload adapter only supports 1d kernel launches at the moment";
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ol_kernel_launch_size_args_t LaunchArgs;
  LaunchArgs.Dimensions = workDim;
  LaunchArgs.NumGroupsX = pGlobalWorkSize[0];
  LaunchArgs.NumGroupsY = 1;
  LaunchArgs.NumGroupsZ = 1;
  LaunchArgs.GroupSizeX = 1;
  LaunchArgs.GroupSizeY = 1;
  LaunchArgs.GroupSizeZ = 1;
  LaunchArgs.DynSharedMemory = 0;

  ol_event_handle_t EventOut;
  auto Ret =
      olLaunchKernel(hQueue->OffloadQueue, hQueue->OffloadDevice,
                     hKernel->OffloadKernel, hKernel->Args.getStorage(),
                     hKernel->Args.getStorageSize(), &LaunchArgs, &EventOut);

  if (Ret != OL_SUCCESS) {
    return offloadResultToUR(Ret);
  }

  if (phEvent) {
    auto *Event = new ur_event_handle_t_();
    Event->OffloadEvent = EventOut;
    *phEvent = Event;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMFill2D(
    ur_queue_handle_t, void *, size_t, size_t, const void *, size_t, size_t,
    uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueUSMMemcpy2D(
    ur_queue_handle_t, bool, void *, size_t, const void *, size_t, size_t,
    size_t, uint32_t, const ur_event_handle_t *, ur_event_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
