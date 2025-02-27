//===--------- command_buffer.cpp - CUDA Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
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

#include <cstring>

namespace {
ur_result_t
commandBufferDestroy(ur_exp_command_buffer_handle_t CommandBuffer) try {
  // Release the memory allocated to the CudaGraph
  UR_CHECK_ERROR(cuGraphDestroy(CommandBuffer->CudaGraph));

  // Release the memory allocated to the CudaGraphExec
  if (CommandBuffer->CudaGraphExec) {
    UR_CHECK_ERROR(cuGraphExecDestroy(CommandBuffer->CudaGraphExec));
  }

  delete CommandBuffer;
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

ur_result_t commandHandleDestroy(
    std::unique_ptr<ur_exp_command_buffer_command_handle_t_> &Command) try {
  // We create the ur_event_t returned to the user for a signal node using
  // `makeWithNative` which sets `HasOwnership` to false. Therefore destruction
  // of the `ur_event_t` object doesn't free the underlying CuEvent_t object and
  // we need to do it manually ourselves.
  if (Command->SignalNode) {
    CUevent SignalEvent{};
    UR_CHECK_ERROR(
        cuGraphEventRecordNodeGetEvent(Command->SignalNode, &SignalEvent));
    UR_CHECK_ERROR(cuEventDestroy(SignalEvent));
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}
} // end anonymous namespace

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t Context, ur_device_handle_t Device, bool IsUpdatable)
    : Context(Context), Device(Device), IsUpdatable(IsUpdatable),
      CudaGraph{nullptr}, CudaGraphExec{nullptr}, RefCount{1},
      NextSyncPoint{0} {
  urContextRetain(Context);
  urDeviceRetain(Device);
}

/// The ur_exp_command_buffer_handle_t_ destructor releases
/// all the memory objects allocated for command_buffer managment
ur_exp_command_buffer_handle_t_::~ur_exp_command_buffer_handle_t_() {
  // Release the memory allocated to the Context stored in the command_buffer
  UR_TRACE(urContextRelease(Context));

  // Release the device
  UR_TRACE(urDeviceRelease(Device));
}

// This may throw so it must be called from within a try...catch
std::unique_ptr<ur_event_handle_t_>
ur_exp_command_buffer_handle_t_::addSignalNode(CUgraphNode DepNode,
                                               CUgraphNode &SignalNode) {
  CUevent Event{};
  UR_CHECK_ERROR(cuEventCreate(&Event, CU_EVENT_DEFAULT));
  UR_CHECK_ERROR(
      cuGraphAddEventRecordNode(&SignalNode, CudaGraph, &DepNode, 1, Event));

  return std::unique_ptr<ur_event_handle_t_>(
      ur_event_handle_t_::makeWithNative(Context, Event));
}

ur_result_t ur_exp_command_buffer_handle_t_::addWaitNodes(
    std::vector<CUgraphNode> &DepsList, uint32_t NumEventsInWaitList,
    const ur_event_handle_t *EventWaitList) try {
  std::vector<CUgraphNode> WaitNodes(NumEventsInWaitList);
  for (uint32_t i = 0; i < NumEventsInWaitList; i++) {
    CUevent Event = EventWaitList[i]->get();
    UR_CHECK_ERROR(cuGraphAddEventWaitNode(
        &WaitNodes[i], CudaGraph, DepsList.data(), DepsList.size(), Event));
  }
  // Set DepsLists as an output parameter for communicating the list of wait
  // nodes created.
  DepsList = std::move(WaitNodes);
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

kernel_command_handle::kernel_command_handle(
    ur_exp_command_buffer_handle_t CommandBuffer, ur_kernel_handle_t Kernel,
    CUgraphNode Node, CUDA_KERNEL_NODE_PARAMS Params, uint32_t WorkDim,
    const size_t *GlobalWorkOffsetPtr, const size_t *GlobalWorkSizePtr,
    const size_t *LocalWorkSizePtr, uint32_t NumKernelAlternatives,
    ur_kernel_handle_t *KernelAlternatives, CUgraphNode SignalNode,
    const std::vector<CUgraphNode> &WaitNodes)
    : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                              WaitNodes),
      Kernel(Kernel), Params(Params), WorkDim(WorkDim) {
  const size_t CopySize = sizeof(size_t) * WorkDim;
  std::memcpy(GlobalWorkOffset, GlobalWorkOffsetPtr, CopySize);
  std::memcpy(GlobalWorkSize, GlobalWorkSizePtr, CopySize);
  // Local work size may be nullptr
  if (LocalWorkSizePtr) {
    std::memcpy(LocalWorkSize, LocalWorkSizePtr, CopySize);
  } else {
    std::memset(LocalWorkSize, 0, sizeof(size_t) * 3);
  }

  if (WorkDim < 3) {
    const size_t ZeroSize = sizeof(size_t) * (3 - WorkDim);
    std::memset(GlobalWorkOffset + WorkDim, 0, ZeroSize);
    std::memset(GlobalWorkSize + WorkDim, 0, ZeroSize);
  }

  /* Add the default Kernel as a valid kernel handle for this command */
  ValidKernelHandles.insert(Kernel);
  if (KernelAlternatives) {
    ValidKernelHandles.insert(KernelAlternatives,
                              KernelAlternatives + NumKernelAlternatives);
  }
};

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

  // For each sync-point add associated CUDA graph node to the return list.
  for (size_t i = 0; i < NumSyncPointsInWaitList; i++) {
    if (auto NodeHandle = SyncPoints.find(SyncPointWaitList[i]);
        NodeHandle != SyncPoints.end()) {
      CuNodesList.push_back(NodeHandle->second);
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
  // Set all params to 0 first
  std::memset(&Params, 0, sizeof(CUDA_MEMCPY3D));

  Params.srcMemoryType = SrcType;
  Params.srcDevice = SrcType == CU_MEMORYTYPE_DEVICE
                         ? *static_cast<const CUdeviceptr *>(SrcPtr)
                         : 0;
  Params.srcHost = SrcType == CU_MEMORYTYPE_HOST ? SrcPtr : nullptr;
  Params.dstMemoryType = DstType;
  Params.dstDevice =
      DstType == CU_MEMORYTYPE_DEVICE ? *static_cast<CUdeviceptr *>(DstPtr) : 0;
  Params.dstHost = DstType == CU_MEMORYTYPE_HOST ? DstPtr : nullptr;
  Params.WidthInBytes = Size;
  Params.Height = 1;
  Params.Depth = 1;
}

// Helper function for enqueuing memory fills. Templated on the CommandType
// enum class for the type of fill being created.
template <class T>
static ur_result_t enqueueCommandBufferFillHelper(
    ur_exp_command_buffer_handle_t CommandBuffer, void *DstDevice,
    const CUmemorytype_enum DstType, const void *Pattern, size_t PatternSize,
    size_t Size, uint32_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    uint32_t NumEventsInWaitList, const ur_event_handle_t *EventWaitList,
    ur_exp_command_buffer_sync_point_t *RetSyncPoint,
    ur_event_handle_t *RetEvent,
    ur_exp_command_buffer_command_handle_t *RetCommand) try {
  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(CommandBuffer, NumSyncPointsInWaitList,
                                        SyncPointWaitList, DepsList));

  if (NumEventsInWaitList) {
    UR_CHECK_ERROR(CommandBuffer->addWaitNodes(DepsList, NumEventsInWaitList,
                                               EventWaitList));
  }

  // CUDA has no memset functions that allow setting values more than 4
  // bytes. UR API lets you pass an arbitrary "pattern" to the buffer
  // fill, which can be more than 4 bytes. Calculate the number of steps
  // required here to see if decomposing to multiple fill nodes is required.
  size_t NumberOfSteps = PatternSize / sizeof(uint8_t);

  // Graph node added to graph, if multiple nodes are created this will
  // be set to the leaf node
  CUgraphNode GraphNode;
  // Track if multiple nodes are created so we can pass them to the command
  // handle
  std::vector<CUgraphNode> DecomposedNodes;

  if (NumberOfSteps > 4) {
    DecomposedNodes.reserve(NumberOfSteps);
  }

  const size_t N = Size / PatternSize;
  auto DstPtr = DstType == CU_MEMORYTYPE_DEVICE
                    ? *static_cast<CUdeviceptr *>(DstDevice)
                    : (CUdeviceptr)DstDevice;

  if (NumberOfSteps <= 4) {
    CUDA_MEMSET_NODE_PARAMS NodeParams = {};
    NodeParams.dst = DstPtr;
    NodeParams.elementSize = PatternSize;
    NodeParams.height = N;
    NodeParams.pitch = PatternSize;
    NodeParams.width = 1;

    // pattern size in bytes
    switch (PatternSize) {
    case 1: {
      auto Value = *static_cast<const uint8_t *>(Pattern);
      NodeParams.value = Value;
      break;
    }
    case 2: {
      auto Value = *static_cast<const uint16_t *>(Pattern);
      NodeParams.value = Value;
      break;
    }
    case 4: {
      auto Value = *static_cast<const uint32_t *>(Pattern);
      NodeParams.value = Value;
      break;
    }
    }

    UR_CHECK_ERROR(cuGraphAddMemsetNode(
        &GraphNode, CommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParams, CommandBuffer->Device->getNativeContext()));
  } else {
    // We must break up the rest of the pattern into 1 byte values, and set
    // the buffer using multiple strided calls. This means that one
    // cuGraphAddMemsetNode call is made for every 1 bytes in the pattern.

    // Update NodeParam
    CUDA_MEMSET_NODE_PARAMS NodeParamsStepFirst = {};
    NodeParamsStepFirst.dst = DstPtr;
    NodeParamsStepFirst.elementSize = sizeof(uint32_t);
    NodeParamsStepFirst.height = Size / sizeof(uint32_t);
    NodeParamsStepFirst.pitch = sizeof(uint32_t);
    NodeParamsStepFirst.value = *static_cast<const uint32_t *>(Pattern);
    NodeParamsStepFirst.width = 1;

    // Inital decomposed node depends on the provided external event wait
    // nodes
    UR_CHECK_ERROR(cuGraphAddMemsetNode(
        &GraphNode, CommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
        &NodeParamsStepFirst, CommandBuffer->Device->getNativeContext()));

    DecomposedNodes.push_back(GraphNode);

    // we walk up the pattern in 1-byte steps, and call cuMemset for each
    // 1-byte chunk of the pattern.
    for (auto Step = 4u; Step < NumberOfSteps; ++Step) {
      // take 4 bytes of the pattern
      auto Value = *(static_cast<const uint8_t *>(Pattern) + Step);

      // offset the pointer to the part of the buffer we want to write to
      auto OffsetPtr = DstPtr + (Step * sizeof(uint8_t));

      // Update NodeParam
      CUDA_MEMSET_NODE_PARAMS NodeParamsStep = {};
      NodeParamsStep.dst = (CUdeviceptr)OffsetPtr;
      NodeParamsStep.elementSize = sizeof(uint8_t);
      NodeParamsStep.height = Size / NumberOfSteps;
      NodeParamsStep.pitch = NumberOfSteps * sizeof(uint8_t);
      NodeParamsStep.value = Value;
      NodeParamsStep.width = 1;

      // Copy the last GraphNode ptr so we can pass it as the dependency for
      // the next one
      CUgraphNode PrevNode = GraphNode;

      UR_CHECK_ERROR(cuGraphAddMemsetNode(
          &GraphNode, CommandBuffer->CudaGraph, &PrevNode, 1, &NodeParamsStep,
          CommandBuffer->Device->getNativeContext()));

      // Store the decomposed node
      DecomposedNodes.push_back(GraphNode);
    }
  }

  CUgraphNode SignalNode = nullptr;
  if (RetEvent) {
    auto SignalEvent = CommandBuffer->addSignalNode(GraphNode, SignalNode);
    *RetEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = CommandBuffer->addSyncPoint(SyncPointNode);
  if (RetSyncPoint) {
    *RetSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      NumEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<T>(CommandBuffer, GraphNode, SignalNode,
                                        WaitNodes, std::move(DecomposedNodes));
  if (RetCommand) {
    *RetCommand = NewCommand.get();
  }

  CommandBuffer->CommandHandles.push_back(std::move(NewCommand));

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    ur_exp_command_buffer_handle_t *phCommandBuffer) {
  const bool IsUpdatable = pCommandBufferDesc->isUpdatable;
  try {
    *phCommandBuffer =
        new ur_exp_command_buffer_handle_t_(hContext, hDevice, IsUpdatable);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  try {
    UR_CHECK_ERROR(cuGraphCreate(&(*phCommandBuffer)->CudaGraph, 0));
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
  if (hCommandBuffer->decrementReferenceCount() == 0) {
    // Ref count has reached zero, release of created commands
    for (auto &Command : hCommandBuffer->CommandHandles) {
      commandHandleDestroy(Command);
    }

    return commandBufferDestroy(hCommandBuffer);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  UR_ASSERT(hCommandBuffer->CudaGraphExec == nullptr,
            UR_RESULT_ERROR_INVALID_OPERATION);
  try {
    const unsigned long long flags = 0;
#if CUDA_VERSION >= 12000
    UR_CHECK_ERROR(cuGraphInstantiate(&hCommandBuffer->CudaGraphExec,
                                      hCommandBuffer->CudaGraph, flags));
#elif CUDA_VERSION >= 11040
    UR_CHECK_ERROR(cuGraphInstantiateWithFlags(
        &hCommandBuffer->CudaGraphExec, hCommandBuffer->CudaGraph, flags));
#else
    // Cannot use flags
    UR_CHECK_ERROR(cuGraphInstantiate(&hCommandBuffer->CudaGraphExec,
                                      hCommandBuffer->CudaGraph, nullptr,
                                      nullptr, 0));
#endif
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numKernelAlternatives, ur_kernel_handle_t *phKernelAlternatives,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) {
  // Preconditions
  // Command handles can only be obtained from updatable command-buffers
  UR_ASSERT(!(phCommand && !hCommandBuffer->IsUpdatable),
            UR_RESULT_ERROR_INVALID_OPERATION);
  UR_ASSERT(hCommandBuffer->Context == hKernel->getContext(),
            UR_RESULT_ERROR_INVALID_KERNEL);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  for (uint32_t i = 0; i < numKernelAlternatives; ++i) {
    UR_ASSERT(phKernelAlternatives[i] != hKernel,
              UR_RESULT_ERROR_INVALID_VALUE);
  }

  try {
    CUgraphNode GraphNode;

    std::vector<CUgraphNode> DepsList;
    UR_CHECK_ERROR(getNodesFromSyncPoints(
        hCommandBuffer, numSyncPointsInWaitList, pSyncPointWaitList, DepsList));

    if (numEventsInWaitList) {
      UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                  phEventWaitList));
    }

    if (*pGlobalWorkSize == 0) {
      // Create an empty node if the kernel workload size is zero
      if (!phEvent) {
        UR_CHECK_ERROR(cuGraphAddEmptyNode(&GraphNode,
                                           hCommandBuffer->CudaGraph,
                                           DepsList.data(), DepsList.size()));
      } else {
        CUevent Event = nullptr;
        UR_CHECK_ERROR(cuEventCreate(&Event, CU_EVENT_DEFAULT));
        UR_CHECK_ERROR(
            cuGraphAddEventRecordNode(&GraphNode, hCommandBuffer->CudaGraph,
                                      DepsList.data(), DepsList.size(), Event));

        auto RetEventUP = std::unique_ptr<ur_event_handle_t_>(
            ur_event_handle_t_::makeWithNative(hCommandBuffer->Context, Event));

        *phEvent = RetEventUP.release();
      }

      // Add signal node if external return event is used.
      CUgraphNode SignalNode = nullptr;
      if (phEvent) {
        auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
        *phEvent = SignalEvent.release();
      }

      // Get sync point and register the cuNode with it.
      CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
      auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
      if (pSyncPoint) {
        *pSyncPoint = SyncPoint;
      }
      return UR_RESULT_SUCCESS;
    }

    // Set the number of threads per block to the number of threads per warp
    // by default unless user has provided a better number
    size_t ThreadsPerBlock[3] = {32u, 1u, 1u};
    size_t BlocksPerGrid[3] = {1u, 1u, 1u};

    uint32_t LocalSize = hKernel->getLocalSize();
    CUfunction CuFunc = hKernel->get();
    UR_CHECK_ERROR(setKernelParams(
        hCommandBuffer->Context, hCommandBuffer->Device, workDim,
        pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize, hKernel, CuFunc,
        ThreadsPerBlock, BlocksPerGrid));

    // Set node param structure with the kernel related data
    auto &ArgPointers = hKernel->getArgPointers();
    CUDA_KERNEL_NODE_PARAMS NodeParams = {};
    NodeParams.func = CuFunc;
    NodeParams.gridDimX = BlocksPerGrid[0];
    NodeParams.gridDimY = BlocksPerGrid[1];
    NodeParams.gridDimZ = BlocksPerGrid[2];
    NodeParams.blockDimX = ThreadsPerBlock[0];
    NodeParams.blockDimY = ThreadsPerBlock[1];
    NodeParams.blockDimZ = ThreadsPerBlock[2];
    NodeParams.sharedMemBytes = LocalSize;
    NodeParams.kernelParams = const_cast<void **>(ArgPointers.data());

    // Create and add an new kernel node to the Cuda graph
    UR_CHECK_ERROR(cuGraphAddKernelNode(&GraphNode, hCommandBuffer->CudaGraph,
                                        DepsList.data(), DepsList.size(),
                                        &NodeParams));

    // Add signal node if external return event is used.
    CUgraphNode SignalNode = nullptr;
    if (phEvent) {
      auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
      *phEvent = SignalEvent.release();
    }

    // Get sync point and register the cuNode with it.
    CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
    auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
    if (pSyncPoint) {
      *pSyncPoint = SyncPoint;
    }

    std::vector<CUgraphNode> WaitNodes =
        numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
    auto NewCommand = std::make_unique<kernel_command_handle>(
        hCommandBuffer, hKernel, GraphNode, NodeParams, workDim,
        pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
        numKernelAlternatives, phKernelAlternatives, SignalNode, WaitNodes);

    if (phCommand) {
      *phCommand = NewCommand.get();
    }

    hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  } catch (ur_result_t Err) {
    return Err;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  CUDA_MEMCPY3D NodeParams = {};
  setCopyParams(pSrc, CU_MEMORYTYPE_HOST, pDst, CU_MEMORYTYPE_HOST, size,
                NodeParams);

  UR_CHECK_ERROR(cuGraphAddMemcpyNode(
      &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
      &NodeParams, hCommandBuffer->Device->getNativeContext()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<usm_memcpy_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);
  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;

  UR_ASSERT(size + dstOffset <= std::get<BufferMem>(hDstMem->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(size + srcOffset <= std::get<BufferMem>(hSrcMem->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  auto Src = std::get<BufferMem>(hSrcMem->Mem)
                 .getPtrWithOffset(hCommandBuffer->Device, srcOffset);
  auto Dst = std::get<BufferMem>(hDstMem->Mem)
                 .getPtrWithOffset(hCommandBuffer->Device, dstOffset);

  CUDA_MEMCPY3D NodeParams = {};
  setCopyParams(&Src, CU_MEMORYTYPE_DEVICE, &Dst, CU_MEMORYTYPE_DEVICE, size,
                NodeParams);

  UR_CHECK_ERROR(cuGraphAddMemcpyNode(
      &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
      &NodeParams, hCommandBuffer->Device->getNativeContext()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<buffer_copy_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);

  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  auto SrcPtr =
      std::get<BufferMem>(hSrcMem->Mem).getPtr(hCommandBuffer->Device);
  auto DstPtr =
      std::get<BufferMem>(hDstMem->Mem).getPtr(hCommandBuffer->Device);
  CUDA_MEMCPY3D NodeParams = {};

  setCopyRectParams(region, &SrcPtr, CU_MEMORYTYPE_DEVICE, srcOrigin,
                    srcRowPitch, srcSlicePitch, &DstPtr, CU_MEMORYTYPE_DEVICE,
                    dstOrigin, dstRowPitch, dstSlicePitch, NodeParams);

  UR_CHECK_ERROR(cuGraphAddMemcpyNode(
      &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
      &NodeParams, hCommandBuffer->Device->getNativeContext()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<buffer_copy_rect_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);

  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  auto Dst = std::get<BufferMem>(hBuffer->Mem)
                 .getPtrWithOffset(hCommandBuffer->Device, offset);

  CUDA_MEMCPY3D NodeParams = {};
  setCopyParams(pSrc, CU_MEMORYTYPE_HOST, &Dst, CU_MEMORYTYPE_DEVICE, size,
                NodeParams);

  UR_CHECK_ERROR(cuGraphAddMemcpyNode(
      &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
      &NodeParams, hCommandBuffer->Device->getNativeContext()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<buffer_write_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);
  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  auto Src = std::get<BufferMem>(hBuffer->Mem)
                 .getPtrWithOffset(hCommandBuffer->Device, offset);

  CUDA_MEMCPY3D NodeParams = {};
  setCopyParams(&Src, CU_MEMORYTYPE_DEVICE, pDst, CU_MEMORYTYPE_HOST, size,
                NodeParams);

  UR_CHECK_ERROR(cuGraphAddMemcpyNode(
      &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
      &NodeParams, hCommandBuffer->Device->getNativeContext()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<buffer_read_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);
  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  auto DstPtr =
      std::get<BufferMem>(hBuffer->Mem).getPtr(hCommandBuffer->Device);
  CUDA_MEMCPY3D NodeParams = {};

  setCopyRectParams(region, pSrc, CU_MEMORYTYPE_HOST, hostOffset, hostRowPitch,
                    hostSlicePitch, &DstPtr, CU_MEMORYTYPE_DEVICE, bufferOffset,
                    bufferRowPitch, bufferSlicePitch, NodeParams);

  UR_CHECK_ERROR(cuGraphAddMemcpyNode(
      &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
      &NodeParams, hCommandBuffer->Device->getNativeContext()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<buffer_write_rect_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);

  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  CUgraphNode GraphNode;
  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  auto SrcPtr =
      std::get<BufferMem>(hBuffer->Mem).getPtr(hCommandBuffer->Device);
  CUDA_MEMCPY3D NodeParams = {};

  setCopyRectParams(region, &SrcPtr, CU_MEMORYTYPE_DEVICE, bufferOffset,
                    bufferRowPitch, bufferSlicePitch, pDst, CU_MEMORYTYPE_HOST,
                    hostOffset, hostRowPitch, hostSlicePitch, NodeParams);

  UR_CHECK_ERROR(cuGraphAddMemcpyNode(
      &GraphNode, hCommandBuffer->CudaGraph, DepsList.data(), DepsList.size(),
      &NodeParams, hCommandBuffer->Device->getNativeContext()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<buffer_read_rect_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);

  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMPrefetchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void * /* Mem */,
    size_t /*Size*/, ur_usm_migration_flags_t /*Flags*/,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  // Prefetch cmd is not supported by Cuda Graph.
  // We implement it as an empty node to enforce dependencies.
  CUgraphNode GraphNode;

  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  // Add an empty node to preserve dependencies.
  UR_CHECK_ERROR(cuGraphAddEmptyNode(&GraphNode, hCommandBuffer->CudaGraph,
                                     DepsList.data(), DepsList.size()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<usm_prefetch_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);

  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMAdviseExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, const void * /* Mem */,
    size_t /*Size*/, ur_usm_advice_flags_t /*Advice*/,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) try {
  // Mem-Advise cmd is not supported by Cuda Graph.
  // We implement it as an empty node to enforce dependencies.
  CUgraphNode GraphNode;

  std::vector<CUgraphNode> DepsList;
  UR_CHECK_ERROR(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                        pSyncPointWaitList, DepsList));

  if (numEventsInWaitList) {
    UR_CHECK_ERROR(hCommandBuffer->addWaitNodes(DepsList, numEventsInWaitList,
                                                phEventWaitList));
  }

  // Add an empty node to preserve dependencies.
  UR_CHECK_ERROR(cuGraphAddEmptyNode(&GraphNode, hCommandBuffer->CudaGraph,
                                     DepsList.data(), DepsList.size()));

  // Add signal node if external return event is used.
  CUgraphNode SignalNode = nullptr;
  if (phEvent) {
    auto SignalEvent = hCommandBuffer->addSignalNode(GraphNode, SignalNode);
    *phEvent = SignalEvent.release();
  }

  // Get sync point and register the cuNode with it.
  CUgraphNode SyncPointNode = SignalNode ? SignalNode : GraphNode;
  auto SyncPoint = hCommandBuffer->addSyncPoint(SyncPointNode);
  if (pSyncPoint) {
    *pSyncPoint = SyncPoint;
  }

  std::vector<CUgraphNode> WaitNodes =
      numEventsInWaitList ? std::move(DepsList) : std::vector<CUgraphNode>();
  auto NewCommand = std::make_unique<usm_advise_command_handle>(
      hCommandBuffer, GraphNode, SignalNode, WaitNodes);

  if (phCommand) {
    *phCommand = NewCommand.get();
  }

  hCommandBuffer->CommandHandles.push_back(std::move(NewCommand));

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    const void *pPattern, size_t patternSize, size_t offset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto ArgsAreMultiplesOfPatternSize =
      (offset % patternSize == 0) || (size % patternSize == 0);

  auto PatternIsValid = (pPattern != nullptr);

  auto PatternSizeIsValid = ((patternSize & (patternSize - 1)) == 0) &&
                            (patternSize > 0); // is a positive power of two
  UR_ASSERT(ArgsAreMultiplesOfPatternSize && PatternIsValid &&
                PatternSizeIsValid,
            UR_RESULT_ERROR_INVALID_SIZE);

  auto DstDevice = std::get<BufferMem>(hBuffer->Mem)
                       .getPtrWithOffset(hCommandBuffer->Device, offset);

  return enqueueCommandBufferFillHelper<buffer_fill_command_handle>(
      hCommandBuffer, &DstDevice, CU_MEMORYTYPE_DEVICE, pPattern, patternSize,
      size, numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMFillExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pPtr,
    const void *pPattern, size_t patternSize, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint, ur_event_handle_t *phEvent,
    ur_exp_command_buffer_command_handle_t *phCommand) {
  auto PatternIsValid = (pPattern != nullptr);

  auto PatternSizeIsValid = ((patternSize & (patternSize - 1)) == 0) &&
                            (patternSize > 0); // is a positive power of two

  UR_ASSERT(PatternIsValid && PatternSizeIsValid, UR_RESULT_ERROR_INVALID_SIZE);
  return enqueueCommandBufferFillHelper<usm_fill_command_handle>(
      hCommandBuffer, pPtr, CU_MEMORYTYPE_UNIFIED, pPattern, patternSize, size,
      numSyncPointsInWaitList, pSyncPointWaitList, numEventsInWaitList,
      phEventWaitList, pSyncPoint, phEvent, phCommand);
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) try {
  std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
  ScopedContext Active(hQueue->getDevice());
  uint32_t StreamToken;
  ur_stream_guard_ Guard;
  CUstream CuStream = hQueue->getNextComputeStream(
      numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

  UR_CHECK_ERROR(enqueueEventsWait(hQueue, CuStream, numEventsInWaitList,
                                   phEventWaitList));

  if (phEvent) {
    RetImplEvent = std::unique_ptr<ur_event_handle_t_>(
        ur_event_handle_t_::makeNative(UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP,
                                       hQueue, CuStream, StreamToken));
    UR_CHECK_ERROR(RetImplEvent->start());
  }

  // Launch graph
  UR_CHECK_ERROR(cuGraphLaunch(hCommandBuffer->CudaGraphExec, CuStream));

  if (phEvent) {
    UR_CHECK_ERROR(RetImplEvent->record());
    *phEvent = RetImplEvent.release();
  }
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

/**
 * Validates contents of the update command description.
 * @param[in] CommandBuffer The command-buffer which is being updated.
 * @param[in] UpdateCommandDesc The update command description.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t
validateCommandDesc(ur_exp_command_buffer_handle_t CommandBuffer,
                    const ur_exp_command_buffer_update_kernel_launch_desc_t
                        &UpdateCommandDesc) {
  if (UpdateCommandDesc.hCommand->getCommandType() != CommandType::Kernel) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  auto Command =
      static_cast<kernel_command_handle *>(UpdateCommandDesc.hCommand);
  if (CommandBuffer != Command->CommandBuffer) {
    return UR_RESULT_ERROR_INVALID_COMMAND_BUFFER_COMMAND_HANDLE_EXP;
  }

  // Update requires the command-buffer to be finalized and updatable.
  if (!CommandBuffer->CudaGraphExec || !CommandBuffer->IsUpdatable) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  if (UpdateCommandDesc.newWorkDim != Command->WorkDim &&
      (!UpdateCommandDesc.pNewGlobalWorkOffset ||
       !UpdateCommandDesc.pNewGlobalWorkSize)) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  if (UpdateCommandDesc.hNewKernel &&
      !Command->ValidKernelHandles.count(UpdateCommandDesc.hNewKernel)) {
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

/**
 * Updates the arguments of a kernel command.
 * @param[in] UpdateCommandDesc The update command description that contains
 * the new configuration.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t
updateKernelArguments(const ur_exp_command_buffer_update_kernel_launch_desc_t
                          &UpdateCommandDesc) {
  auto Command =
      static_cast<kernel_command_handle *>(UpdateCommandDesc.hCommand);
  ur_kernel_handle_t Kernel = Command->Kernel;
  ur_device_handle_t Device = Command->CommandBuffer->Device;

  // Update pointer arguments to the kernel
  uint32_t NumPointerArgs = UpdateCommandDesc.numNewPointerArgs;
  const ur_exp_command_buffer_update_pointer_arg_desc_t *ArgPointerList =
      UpdateCommandDesc.pNewPointerArgList;
  for (uint32_t i = 0; i < NumPointerArgs; i++) {
    const auto &PointerArgDesc = ArgPointerList[i];
    uint32_t ArgIndex = PointerArgDesc.argIndex;
    const void *ArgValue = PointerArgDesc.pNewPointerArg;

    ur_result_t Result = UR_RESULT_SUCCESS;
    try {
      Kernel->setKernelArg(ArgIndex, sizeof(ArgValue), ArgValue);
    } catch (ur_result_t Err) {
      Result = Err;
      return Result;
    }
  }

  // Update memobj arguments to the kernel
  uint32_t NumMemobjArgs = UpdateCommandDesc.numNewMemObjArgs;
  const ur_exp_command_buffer_update_memobj_arg_desc_t *ArgMemobjList =
      UpdateCommandDesc.pNewMemObjArgList;
  for (uint32_t i = 0; i < NumMemobjArgs; i++) {
    const auto &MemobjArgDesc = ArgMemobjList[i];
    uint32_t ArgIndex = MemobjArgDesc.argIndex;
    ur_mem_handle_t ArgValue = MemobjArgDesc.hNewMemObjArg;

    ur_result_t Result = UR_RESULT_SUCCESS;
    try {
      if (ArgValue == nullptr) {
        Kernel->setKernelArg(ArgIndex, 0, nullptr);
      } else {
        CUdeviceptr CuPtr = std::get<BufferMem>(ArgValue->Mem).getPtr(Device);
        Kernel->setKernelArg(ArgIndex, sizeof(CUdeviceptr), (void *)&CuPtr);
      }
    } catch (ur_result_t Err) {
      Result = Err;
      return Result;
    }
  }

  // Update value arguments to the kernel
  uint32_t NumValueArgs = UpdateCommandDesc.numNewValueArgs;
  const ur_exp_command_buffer_update_value_arg_desc_t *ArgValueList =
      UpdateCommandDesc.pNewValueArgList;
  for (uint32_t i = 0; i < NumValueArgs; i++) {
    const auto &ValueArgDesc = ArgValueList[i];
    uint32_t ArgIndex = ValueArgDesc.argIndex;
    size_t ArgSize = ValueArgDesc.argSize;
    const void *ArgValue = ValueArgDesc.pNewValueArg;

    ur_result_t Result = UR_RESULT_SUCCESS;
    try {
      // Local memory args are passed as value args with nullptr value
      if (ArgValue) {
        Kernel->setKernelArg(ArgIndex, ArgSize, ArgValue);
      } else {
        Kernel->setKernelLocalArg(ArgIndex, ArgSize);
      }
    } catch (ur_result_t Err) {
      Result = Err;
      return Result;
    }
  }

  return UR_RESULT_SUCCESS;
}

/**
 * Updates the command-buffer command with new values from the update
 * description.
 * @param[in] UpdateCommandDesc The update command description.
 * @return UR_RESULT_SUCCESS or an error code on failure
 */
ur_result_t
updateCommand(const ur_exp_command_buffer_update_kernel_launch_desc_t
                  &UpdateCommandDesc) {
  auto Command =
      static_cast<kernel_command_handle *>(UpdateCommandDesc.hCommand);
  if (UpdateCommandDesc.hNewKernel) {
    Command->Kernel = UpdateCommandDesc.hNewKernel;
  }

  if (UpdateCommandDesc.newWorkDim) {
    Command->WorkDim = UpdateCommandDesc.newWorkDim;
  }

  if (UpdateCommandDesc.pNewGlobalWorkOffset) {
    Command->setGlobalOffset(UpdateCommandDesc.pNewGlobalWorkOffset);
  }

  if (UpdateCommandDesc.pNewGlobalWorkSize) {
    Command->setGlobalSize(UpdateCommandDesc.pNewGlobalWorkSize);
    if (!UpdateCommandDesc.pNewLocalWorkSize) {
      Command->setNullLocalSize();
    }
  }

  if (UpdateCommandDesc.pNewLocalWorkSize) {
    Command->setLocalSize(UpdateCommandDesc.pNewLocalWorkSize);
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, uint32_t numKernelUpdates,
    const ur_exp_command_buffer_update_kernel_launch_desc_t
        *pUpdateKernelLaunch) try {
  // First validate user inputs, as no update should be propagated if there
  // are errors.
  for (uint32_t i = 0; i < numKernelUpdates; i++) {
    UR_CHECK_ERROR(validateCommandDesc(hCommandBuffer, pUpdateKernelLaunch[i]));
  }

  // Store changes in config struct in command handle object
  for (uint32_t i = 0; i < numKernelUpdates; i++) {
    UR_CHECK_ERROR(updateCommand(pUpdateKernelLaunch[i]));
    UR_CHECK_ERROR(updateKernelArguments(pUpdateKernelLaunch[i]));
  }

  // Propagate changes to CUDA driver API
  for (uint32_t i = 0; i < numKernelUpdates; i++) {
    const auto &UpdateCommandDesc = pUpdateKernelLaunch[i];

    // If no work-size is provided make sure we pass nullptr to setKernelParams
    // so it can guess the local work size.
    auto KernelCommandHandle =
        static_cast<kernel_command_handle *>(UpdateCommandDesc.hCommand);
    const bool ProvidedLocalSize = !KernelCommandHandle->isNullLocalSize();
    size_t *LocalWorkSize =
        ProvidedLocalSize ? KernelCommandHandle->LocalWorkSize : nullptr;

    // Set the number of threads per block to the number of threads per warp
    // by default unless user has provided a better number.
    size_t ThreadsPerBlock[3] = {32u, 1u, 1u};
    size_t BlocksPerGrid[3] = {1u, 1u, 1u};
    CUfunction CuFunc = KernelCommandHandle->Kernel->get();
    auto Result = setKernelParams(
        hCommandBuffer->Context, hCommandBuffer->Device,
        KernelCommandHandle->WorkDim, KernelCommandHandle->GlobalWorkOffset,
        KernelCommandHandle->GlobalWorkSize, LocalWorkSize,
        KernelCommandHandle->Kernel, CuFunc, ThreadsPerBlock, BlocksPerGrid);
    if (Result != UR_RESULT_SUCCESS) {
      return Result;
    }

    CUDA_KERNEL_NODE_PARAMS &Params = KernelCommandHandle->Params;

    Params.func = CuFunc;
    Params.gridDimX = BlocksPerGrid[0];
    Params.gridDimY = BlocksPerGrid[1];
    Params.gridDimZ = BlocksPerGrid[2];
    Params.blockDimX = ThreadsPerBlock[0];
    Params.blockDimY = ThreadsPerBlock[1];
    Params.blockDimZ = ThreadsPerBlock[2];
    Params.sharedMemBytes = KernelCommandHandle->Kernel->getLocalSize();
    Params.kernelParams = const_cast<void **>(
        KernelCommandHandle->Kernel->getArgPointers().data());

    CUgraphNode Node = KernelCommandHandle->Node;
    CUgraphExec CudaGraphExec = hCommandBuffer->CudaGraphExec;
    UR_CHECK_ERROR(
        cuGraphExecKernelNodeSetParams(CudaGraphExec, Node, &Params));
  }
  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateSignalEventExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    ur_event_handle_t *phEvent) try {
  ur_exp_command_buffer_handle_t CommandBuffer = hCommand->CommandBuffer;

  // Update requires command-buffer to be finalized
  if (!CommandBuffer->CudaGraphExec) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  // Update requires command-buffer to be created with update enabled
  if (!CommandBuffer->IsUpdatable) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  // Error to try to update the signal event, when a signal event wasn't set
  // on creation
  CUgraphNode SignalNode = hCommand->SignalNode;
  if (phEvent != nullptr && SignalNode == nullptr) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  CUevent SignalEvent{};
  UR_CHECK_ERROR(cuGraphEventRecordNodeGetEvent(SignalNode, &SignalEvent));

  if (phEvent) {
    *phEvent = std::unique_ptr<ur_event_handle_t_>(
                   ur_event_handle_t_::makeWithNative(CommandBuffer->Context,
                                                      SignalEvent))
                   .release();
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferUpdateWaitEventsExp(
    ur_exp_command_buffer_command_handle_t hCommand,
    uint32_t NumEventsInWaitList,
    const ur_event_handle_t *phEventWaitList) try {
  ur_exp_command_buffer_handle_t CommandBuffer = hCommand->CommandBuffer;

  // Update requires command-buffer to be finalized
  if (!CommandBuffer->CudaGraphExec) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  // Update requires command-buffer to be created with update enabled
  if (!CommandBuffer->IsUpdatable) {
    return UR_RESULT_ERROR_INVALID_OPERATION;
  }

  // Error if number of wait nodes is not the same as when node was created
  std::vector<CUgraphNode> &WaitNodes = hCommand->WaitNodes;
  if (NumEventsInWaitList != WaitNodes.size()) {
    return UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST;
  }

  CUgraphExec CudaGraphExec = CommandBuffer->CudaGraphExec;
  for (uint32_t i = 0; i < NumEventsInWaitList; i++) {
    ur_event_handle_t WaitEvent = phEventWaitList[i];
    UR_CHECK_ERROR(cuGraphExecEventWaitNodeSetEvent(CudaGraphExec, WaitNodes[i],
                                                    WaitEvent->get()));
  }

  return UR_RESULT_SUCCESS;
} catch (ur_result_t Err) {
  return Err;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferGetInfoExp(
    ur_exp_command_buffer_handle_t hCommandBuffer,
    ur_exp_command_buffer_info_t propName, size_t propSize, void *pPropValue,
    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_EXP_COMMAND_BUFFER_INFO_REFERENCE_COUNT:
    return ReturnValue(hCommandBuffer->getReferenceCount());
  case UR_EXP_COMMAND_BUFFER_INFO_DESCRIPTOR: {
    ur_exp_command_buffer_desc_t Descriptor{};
    Descriptor.stype = UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC;
    Descriptor.pNext = nullptr;
    Descriptor.isUpdatable = hCommandBuffer->IsUpdatable;
    Descriptor.isInOrder = false;
    Descriptor.enableProfiling = false;

    return ReturnValue(Descriptor);
  }
  default:
    assert(false && "Command-buffer info request not implemented");
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}
