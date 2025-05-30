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

  // TODO: We default to 1, 1, 1 here. In future if pLocalWorkSize is not
  // specified, we should pick the "best" one
  size_t WorkSize[3] = {1, 1, 1};
  if (pLocalWorkSize) {
    for (uint32_t I = 0; I < workDim; I++) {
      WorkSize[I] = pLocalWorkSize[I];
    }
  }

  if (WorkSize[0] > pGlobalWorkSize[0] || WorkSize[1] > pGlobalWorkSize[1] ||
      WorkSize[2] > pGlobalWorkSize[2]) {
    return UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE;
  }

  ol_kernel_launch_size_args_t LaunchArgs;
  LaunchArgs.Dimensions = workDim;
  LaunchArgs.NumGroupsX = pGlobalWorkSize[0] / WorkSize[0];
  LaunchArgs.NumGroupsY = pGlobalWorkSize[1] / WorkSize[1];
  LaunchArgs.NumGroupsZ = pGlobalWorkSize[2] / WorkSize[2];
  LaunchArgs.GroupSizeX = WorkSize[0];
  LaunchArgs.GroupSizeY = WorkSize[1];
  LaunchArgs.GroupSizeZ = WorkSize[2];
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
