//===--------- command_buffer.cpp - CUDA Adapter --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer.hpp"
#include "common.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t hContext, ur_device_handle_t hDevice)
    : Context(hContext), Device(hDevice), CudaGraph{nullptr},
      CudaGraphExec{nullptr}, RefCount{1} {
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

/// Set parameter for General 1D memory copy.
/// If the source and/or destination is on the device, SrcPtr and/or DstPtr
/// must be a pointer to a CUdeviceptr
static void setCopyParams(const void *SrcPtr, const CUmemorytype_enum SrcType,
                          void *DstPtr, const CUmemorytype_enum DstType,
                          size_t Size, CUDA_MEMCPY3D &Params) {

  Params.srcMemoryType = SrcType;
  Params.srcDevice = SrcType == CU_MEMORYTYPE_DEVICE
                         ? *static_cast<const CUdeviceptr *>(SrcPtr)
                         : 0;
  Params.srcHost = SrcType == CU_MEMORYTYPE_HOST ? SrcPtr : nullptr;
  Params.srcXInBytes = 0;
  Params.srcY = 0;
  Params.srcZ = 0;
  Params.srcLOD = 0;
  Params.srcPitch = 0;
  Params.srcHeight = 0;

  Params.dstMemoryType = DstType;
  Params.dstDevice =
      DstType == CU_MEMORYTYPE_DEVICE ? *static_cast<CUdeviceptr *>(DstPtr) : 0;
  Params.dstHost = DstType == CU_MEMORYTYPE_HOST ? DstPtr : nullptr;
  Params.dstXInBytes = 0;
  Params.dstY = 0;
  Params.dstZ = 0;
  Params.dstLOD = 0;
  Params.dstPitch = 0;
  Params.dstHeight = 0;

  Params.WidthInBytes = Size;
  Params.Height = 1;
  Params.Depth = 1;
}

// Helper function for enqueuing memory fills
static ur_result_t enqueueCommandBufferFillHelper(
    ur_exp_command_buffer_handle_t CommandBuffer, void *DstDevice,
    const CUmemorytype_enum DstType, const void *Pattern, size_t PatternSize,
    size_t Size, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *SyncPoint) {
  ur_result_t Result;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                 SyncPointWaitList, DepsList));

  try {
    size_t N = Size / PatternSize;
    auto Value = *static_cast<const uint32_t *>(Pattern);
    auto DstPtr = DstType == CU_MEMORYTYPE_DEVICE
                      ? *static_cast<CUdeviceptr *>(DstDevice)
                      : (CUdeviceptr)DstDevice;

    if ((PatternSize == 1) || (PatternSize == 2) || (PatternSize == 4)) {
      // Create a new node
      CUgraphNode GraphNode;
      CUDA_MEMSET_NODE_PARAMS NodeParams = {};
      NodeParams.dst = DstPtr;
      NodeParams.elementSize = PatternSize;
      NodeParams.height = N;
      NodeParams.pitch = PatternSize;
      NodeParams.value = Value;
      NodeParams.width = 1;

      Result = UR_CHECK_ERROR(cuGraphAddMemsetNode(
          &GraphNode, CommandBuffer->CudaGraph, DepsList.data(),
          DepsList.size(), &NodeParams, CommandBuffer->Device->getContext()));

      // Get sync point and register the cuNode with it.
      *SyncPoint =
          CommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));

    } else {
      // CUDA has no memset functions that allow setting values more than 4
      // bytes. UR API lets you pass an arbitrary "pattern" to the buffer
      // fill, which can be more than 4 bytes. We must break up the pattern
      // into 4 byte values, and set the buffer using multiple strided calls.
      // This means that one cuGraphAddMemsetNode call is made for every 4 bytes
      // in the pattern.

      size_t NumberOfSteps = PatternSize / sizeof(uint32_t);

      // we walk up the pattern in 4-byte steps, and call cuMemset for each
      // 4-byte chunk of the pattern.
      for (auto Step = 0u; Step < NumberOfSteps; ++Step) {
        // take 4 bytes of the pattern
        auto Value = *(static_cast<const uint32_t *>(Pattern) + Step);

        // offset the pointer to the part of the buffer we want to write to
        auto OffsetPtr = DstPtr + (Step * sizeof(uint32_t));

        // Create a new node
        CUgraphNode GraphNode;
        // Update NodeParam
        CUDA_MEMSET_NODE_PARAMS NodeParamsStep = {};
        NodeParamsStep.dst = (CUdeviceptr)OffsetPtr;
        NodeParamsStep.elementSize = 4;
        NodeParamsStep.height = N;
        NodeParamsStep.pitch = PatternSize;
        NodeParamsStep.value = Value;
        NodeParamsStep.width = 1;

        Result = UR_CHECK_ERROR(cuGraphAddMemsetNode(
            &GraphNode, CommandBuffer->CudaGraph, DepsList.data(),
            DepsList.size(), &NodeParamsStep,
            CommandBuffer->Device->getContext()));

        // Get sync point and register the cuNode with it.
        *SyncPoint = CommandBuffer->AddSyncPoint(
            std::make_shared<CUgraphNode>(GraphNode));
      }
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
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

      // Get sync point and register the cuNode with it.
      *pSyncPoint = hCommandBuffer->AddSyncPoint(
          std::make_shared<CUgraphNode>(GraphNode));
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
    CUDA_KERNEL_NODE_PARAMS NodeParams;
    NodeParams.func = CuFunc;
    NodeParams.gridDimX = BlocksPerGrid[0];
    NodeParams.gridDimY = BlocksPerGrid[1];
    NodeParams.gridDimZ = BlocksPerGrid[2];
    NodeParams.blockDimX = ThreadsPerBlock[0];
    NodeParams.blockDimY = ThreadsPerBlock[1];
    NodeParams.blockDimZ = ThreadsPerBlock[2];
    NodeParams.sharedMemBytes = LocalSize;
    NodeParams.kernelParams = const_cast<void **>(ArgIndices.data());
    NodeParams.extra = nullptr;

    // Create and add an new kernel node to the Cuda graph
    Result = UR_CHECK_ERROR(
        cuGraphAddKernelNode(&GraphNode, hCommandBuffer->CudaGraph,
                             DepsList.data(), DepsList.size(), &NodeParams));

    if (LocalSize != 0)
      hKernel->clearLocalSize();

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result;
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  try {
    CUDA_MEMCPY3D NodeParams = {};
    setCopyParams(pSrc, CU_MEMORYTYPE_HOST, pDst, CU_MEMORYTYPE_HOST, size,
                  NodeParams);

    Result = UR_CHECK_ERROR(cuGraphAddMemcpyNode(
        &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, hCommandBuffer->Device->getContext()));

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result;
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  try {
    auto Src = hSrcMem->Mem.BufferMem.get() + srcOffset;
    auto Dst = hDstMem->Mem.BufferMem.get() + dstOffset;

    CUDA_MEMCPY3D NodeParams = {};
    setCopyParams(&Src, CU_MEMORYTYPE_DEVICE, &Dst, CU_MEMORYTYPE_DEVICE, size,
                  NodeParams);

    Result = UR_CHECK_ERROR(cuGraphAddMemcpyNode(
        &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, hCommandBuffer->Device->getContext()));

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result;
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  try {
    CUdeviceptr SrcPtr = hSrcMem->Mem.BufferMem.get();
    CUdeviceptr DstPtr = hDstMem->Mem.BufferMem.get();
    CUDA_MEMCPY3D NodeParams = {};

    setCopyRectParams(region, &SrcPtr, CU_MEMORYTYPE_DEVICE, srcOrigin,
                      srcRowPitch, srcSlicePitch, &DstPtr, CU_MEMORYTYPE_DEVICE,
                      dstOrigin, dstRowPitch, dstSlicePitch, NodeParams);

    Result = UR_CHECK_ERROR(cuGraphAddMemcpyNode(
        &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, hCommandBuffer->Device->getContext()));

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result;
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  try {
    auto Dst = hBuffer->Mem.BufferMem.get() + offset;

    CUDA_MEMCPY3D NodeParams = {};
    setCopyParams(pSrc, CU_MEMORYTYPE_HOST, &Dst, CU_MEMORYTYPE_DEVICE, size,
                  NodeParams);

    Result = UR_CHECK_ERROR(cuGraphAddMemcpyNode(
        &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, hCommandBuffer->Device->getContext()));

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result;
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  try {
    auto Src = hBuffer->Mem.BufferMem.get() + offset;

    CUDA_MEMCPY3D NodeParams = {};
    setCopyParams(&Src, CU_MEMORYTYPE_DEVICE, pDst, CU_MEMORYTYPE_HOST, size,
                  NodeParams);

    Result = UR_CHECK_ERROR(cuGraphAddMemcpyNode(
        &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, hCommandBuffer->Device->getContext()));

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result;
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  try {
    CUdeviceptr DstPtr = hBuffer->Mem.BufferMem.get();
    CUDA_MEMCPY3D NodeParams = {};

    setCopyRectParams(region, pSrc, CU_MEMORYTYPE_HOST, hostOffset,
                      hostRowPitch, hostSlicePitch, &DstPtr,
                      CU_MEMORYTYPE_DEVICE, bufferOffset, bufferRowPitch,
                      bufferSlicePitch, NodeParams);

    Result = UR_CHECK_ERROR(cuGraphAddMemcpyNode(
        &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, hCommandBuffer->Device->getContext()));

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result;
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList));

  try {
    CUdeviceptr SrcPtr = hBuffer->Mem.BufferMem.get();
    CUDA_MEMCPY3D NodeParams = {};

    setCopyRectParams(region, &SrcPtr, CU_MEMORYTYPE_DEVICE, bufferOffset,
                      bufferRowPitch, bufferSlicePitch, pDst,
                      CU_MEMORYTYPE_HOST, hostOffset, hostRowPitch,
                      hostSlicePitch, NodeParams);

    Result = UR_CHECK_ERROR(cuGraphAddMemcpyNode(
        &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, hCommandBuffer->Device->getContext()));

    // Get sync point and register the cuNode with it.
    *pSyncPoint =
        hCommandBuffer->AddSyncPoint(std::make_shared<CUgraphNode>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    const void *pPattern, size_t patternSize, size_t offset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  auto ArgsAreMultiplesOfPatternSize =
      (offset % patternSize == 0) || (size % patternSize == 0);

  auto PatternIsValid = (pPattern != nullptr);

  auto PatternSizeIsValid = ((patternSize & (patternSize - 1)) == 0) &&
                            (patternSize > 0); // is a positive power of two
  UR_ASSERT(ArgsAreMultiplesOfPatternSize && PatternIsValid &&
                PatternSizeIsValid,
            UR_RESULT_ERROR_INVALID_SIZE);

  auto DstDevice = hBuffer->Mem.BufferMem.get() + offset;

  return enqueueCommandBufferFillHelper(
      hCommandBuffer, &DstDevice, CU_MEMORYTYPE_DEVICE, pPattern, patternSize,
      size, numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pPtr,
    const void *pPattern, size_t patternSize, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {

  auto PatternIsValid = (pPattern != nullptr);

  auto PatternSizeIsValid = ((patternSize & (patternSize - 1)) == 0) &&
                            (patternSize > 0); // is a positive power of two

  UR_ASSERT(PatternIsValid && PatternSizeIsValid, UR_RESULT_ERROR_INVALID_SIZE);
  return enqueueCommandBufferFillHelper(
      hCommandBuffer, pPtr, CU_MEMORYTYPE_UNIFIED, pPattern, patternSize, size,
      numSyncPointsInWaitList, pSyncPointWaitList, pSyncPoint);
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
