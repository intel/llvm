//===--------- launch_attributes.cpp - CUDA Adapter------------------------===//
//
// Copyright (C) 202 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urEnqueueKernelLaunchCustomExp(
    ur_queue_handle_t hQueue, ur_kernel_handle_t hKernel, uint32_t workDim,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numAttrsInLaunchAttrList,
    const ur_exp_launch_attribute_t *launchAttrList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  // Preconditions
  UR_ASSERT(hQueue->getContext() == hKernel->getContext(),
            UR_RESULT_ERROR_INVALID_KERNEL);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  // todo just call regular kernel

  if (launchAttrList == NULL) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  CUlaunchAttribute launch_attribute[numAttrsInLaunchAttrList];
  for (int i = 0; i < numAttrsInLaunchAttrList; i++) {
    switch (launchAttrList[i].id) {
    case LAUNCH_ATTRIBUTE_IGNORE: {
      launch_attribute[i].id = CU_LAUNCH_ATTRIBUTE_IGNORE;
      break;
    }
    case LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION: {

      launch_attribute[i].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launch_attribute[i].value.clusterDim.x =
          launchAttrList[i].value.clusterDim[0];
      launch_attribute[i].value.clusterDim.y =
          launchAttrList[i].value.clusterDim[1];
      launch_attribute[i].value.clusterDim.z =
          launchAttrList[i].value.clusterDim[2];
      break;
    }
    case LAUNCH_ATTRIBUTE_COOPERATIVE: {
      launch_attribute[i].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
      launch_attribute[i].value.cooperative =
          launchAttrList[i].value.cooperative;
      break;
    }
    default: {
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
    }
  }

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

  Result = setKernelParams(hQueue->getContext(), hQueue->Device, workDim, null,
                           pGlobalWorkSize, pLocalWorkSize, hKernel, CuFunc,
                           ThreadsPerBlock, BlocksPerGrid);
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

    CUlaunchConfig launch_config;
    launch_config.gridDimX = grid_dims.x;
    launch_config.gridDimY = grid_dims.y;
    launch_config.gridDimZ = grid_dims.z;
    launch_config.blockDimX = block_dims.x;
    launch_config.blockDimY = block_dims.y;
    launch_config.blockDimZ = block_dims.z;

    launch_config.sharedMemBytes = LocalSize;
    launch_config.hStream = CuStream;
    launch_config.attrs = launch_attribute;
    launch_config.numAttrs = numAttrsInLaunchAttrList;

    UR_CHECK_ERROR(cuLaunchKernelEx(&launch_config, CuFunc,
                                    const_cast<void **>(ArgIndices.data()),
                                    nullptr));

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
