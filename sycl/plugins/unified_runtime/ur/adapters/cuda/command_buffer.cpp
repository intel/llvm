//===--------- command_buffer.cpp - CUDA Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "command_buffer.hpp"
#include "common.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t hContext, ur_device_handle_t hDevice)
    : Context(hContext),
      Device(hDevice), CudaGraph{nullptr}, CudaGraphExec{nullptr}, RefCount{1} {
  urContextRetain(hContext);
  urDeviceRetain(hDevice);
}

// The ur_exp_command_buffer_handle_t_ destructor release all the memory objects
// allocated for command_buffer managment
ur_exp_command_buffer_handle_t_::~ur_exp_command_buffer_handle_t_() {
  // Release the memory allocated to the Context stored in the command_buffer
  urContextRelease(Context);

  // Release the device
  urDeviceRelease(Device);

  // Release the memory allocated to the CudaGraph
  cuGraphDestroy(CudaGraph);

  // Release the memory allocated to the CudaGraphExec
  cuGraphExecDestroy(CudaGraphExec);
}

/// Helper function for finding the Cuda Nodes associated with the
/// commands in a command-buffer, each event is pointed to by a sync-point in
/// the wait list.
///
/// @param[in] CommandBuffer to lookup the events from.
/// @param[in] NumSyncPointsInWaitList Length of \p SyncPointWaitList.
/// @param[in] SyncPointWaitList List of sync points in \p CommandBuffer
/// to find the events for.
/// @param[out] CuNodesList Return parameter for the Cuda Nodes associated with
/// each sync-point in \p SyncPointWaitList.
///
/// @return UR_RESULT_SUCCESS or an error code on failure
static ur_result_t getNodesFromSyncPoints(
    const ur_exp_command_buffer_handle_t &CommandBuffer,
    size_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    std::vector<CUgraphNode> &CuNodesList) {
  // Map of ur_exp_command_buffer_sync_point_t to ur_event_handle_t defining
  // the event associated with each sync-point
  auto SyncPoints = CommandBuffer->SyncPoints;

  // For each sync-point add associated L0 event to the return list.
  for (size_t i = 0; i < NumSyncPointsInWaitList; i++) {
    if (auto NodeHandle = SyncPoints.find(SyncPointWaitList[i]);
        NodeHandle != SyncPoints.end()) {
      CuNodesList.push_back(*NodeHandle->second.get());
    } else {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_exp_command_buffer_desc_t *hCommandBufferDesc,
    ur_exp_command_buffer_handle_t *hCommandBuffer) {
  (void)hCommandBufferDesc;

  try {
    *hCommandBuffer = new ur_exp_command_buffer_handle_t_(hContext, hDevice);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  auto RetCommandBuffer = *hCommandBuffer;
  try {
    UR_CHECK_ERROR(cuGraphCreate(&RetCommandBuffer->CudaGraph, 0));
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  hCommandBuffer->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  if (!hCommandBuffer->decrementAndTestReferenceCount())
    return UR_RESULT_SUCCESS;

  delete hCommandBuffer;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  try {
    UR_CHECK_ERROR(cuGraphInstantiate(&hCommandBuffer->CudaGraphExec,
                                      hCommandBuffer->CudaGraph, 0));
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  // Preconditions
  UR_ASSERT(hCommandBuffer->Context == hKernel->getContext(),
            UR_RESULT_ERROR_INVALID_KERNEL);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ur_result_t Result = UR_RESULT_SUCCESS;
  CUgraphNode GraphNode;

  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  if (*pGlobalWorkSize == 0) {
    try {
      // Create a empty node if the kernel worload size is zero
      Result = UR_CHECK_ERROR(
          cuGraphAddEmptyNode(&GraphNode, hCommandBuffer->CudaGraph,
                              DepsList.data(), DepsList.size()));

      // Get sync point and register the event with it.
      *pSyncPoint = hCommandBuffer->GetNextSyncPoint();
      hCommandBuffer->RegisterSyncPoint(
          *pSyncPoint, std::make_shared<CUgraphNode>(GraphNode));
    } catch (ur_result_t Err) {
      Result = Err;
    }
    return Result;
  }

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  size_t ThreadsPerBlock[3] = {32u, 1u, 1u};
  size_t BlocksPerGrid[3] = {1u, 1u, 1u};

  uint32_t LocalSize = hKernel->getLocalSize();
  CUfunction CuFunc = hKernel->get();

  if ((Result = setKernelParams(
           hCommandBuffer->Context, hCommandBuffer->Device, workDim,
           pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, hKernel, CuFunc,
           ThreadsPerBlock, BlocksPerGrid)) != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    // Set node param structure with the kernel related data
    auto &ArgIndices = hKernel->getArgIndices();
    CUDA_KERNEL_NODE_PARAMS nodeParams;
    nodeParams.func = CuFunc;
    nodeParams.gridDimX = BlocksPerGrid[0];
    nodeParams.gridDimY = BlocksPerGrid[1];
    nodeParams.gridDimZ = BlocksPerGrid[2];
    nodeParams.blockDimX = ThreadsPerBlock[0];
    nodeParams.blockDimY = ThreadsPerBlock[1];
    nodeParams.blockDimZ = ThreadsPerBlock[2];
    nodeParams.sharedMemBytes = LocalSize;
    nodeParams.kernelParams = const_cast<void **>(ArgIndices.data());
    nodeParams.extra = nullptr;

    // Create and add an new kernel node to the Cuda graph
    Result = UR_CHECK_ERROR(
        cuGraphAddKernelNode(&GraphNode, hCommandBuffer->CudaGraph,
                             DepsList.data(), DepsList.size(), &nodeParams));

    if (LocalSize != 0)
      hKernel->clearLocalSize();

    // Get sync point and register the event with it.
    *pSyncPoint = hCommandBuffer->GetNextSyncPoint();
    hCommandBuffer->RegisterSyncPoint(*pSyncPoint,
                                      std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemcpyUSMExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)pDst;
  (void)pSrc;
  (void)size;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for CUDA adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMembufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)hSrcMem;
  (void)hDstMem;
  (void)srcOffset;
  (void)dstOffset;
  (void)size;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for CUDA adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMembufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)hSrcMem;
  (void)hDstMem;
  (void)srcOrigin;
  (void)dstOrigin;
  (void)region;
  (void)srcRowPitch;
  (void)srcSlicePitch;
  (void)dstRowPitch;
  (void)dstSlicePitch;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for CUDA adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)hBuffer;
  (void)offset;
  (void)size;
  (void)pSrc;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for CUDA adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)hBuffer;
  (void)offset;
  (void)size;
  (void)pDst;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for CUDA adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)hBuffer;
  (void)bufferOffset;
  (void)hostOffset;
  (void)region;
  (void)bufferRowPitch;
  (void)bufferSlicePitch;
  (void)hostRowPitch;
  (void)hostSlicePitch;
  (void)pSrc;
  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for CUDA adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMembufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  (void)hCommandBuffer;
  (void)hBuffer;
  (void)bufferOffset;
  (void)hostOffset;
  (void)region;
  (void)bufferRowPitch;
  (void)bufferSlicePitch;
  (void)hostRowPitch;
  (void)hostSlicePitch;
  (void)pDst;

  (void)numSyncPointsInWaitList;
  (void)pSyncPointWaitList;
  (void)pSyncPoint;

  detail::ur::die("Experimental Command-buffer feature is not "
                  "implemented for CUDA adapter.");
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    ScopedContext Active(hQueue->getContext());
    uint32_t StreamToken;
    ur_stream_guard_ Guard;
    CUstream CuStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

    if ((Result = enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                                    phEventWaitList)) != UR_RESULT_SUCCESS) {
      return Result;
    }

    if (phEvent) {
      RetImplEvent =
          std::unique_ptr<ur_event_handle_t_>(ur_event_handle_t_::makeNative(
              UR_COMMAND_KERNEL_LAUNCH, hQueue, CuStream, StreamToken));
      RetImplEvent->start();
    }

    // Launch graph
    Result =
        UR_CHECK_ERROR(cuGraphLaunch(hCommandBuffer->CudaGraphExec, CuStream));

    if (phEvent) {
      Result = RetImplEvent->record();
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}
