//===--------- command_buffer.hpp - CUDA Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ur_print.hpp>

#include "context.hpp"
#include "logger/ur_logger.hpp"
#include <cuda.h>
#include <memory>
#include <unordered_set>

enum class CommandType {
  Kernel,
  USMMemcpy,
  USMFill,
  MemBufferCopy,
  MemBufferCopyRect,
  MemBufferRead,
  MemBufferReadRect,
  MemBufferWrite,
  MemBufferWriteRect,
  MemBufferFill,
  USMPrefetch,
  USMAdvise
};

struct null_command_data {};

struct kernel_command_data {
  kernel_command_data(ur_kernel_handle_t Kernel, CUDA_KERNEL_NODE_PARAMS Params,
                      uint32_t WorkDim, const size_t *GlobalWorkOffsetPtr,
                      const size_t *GlobalWorkSizePtr,
                      const size_t *LocalWorkSizePtr,
                      uint32_t NumKernelAlternatives,
                      ur_kernel_handle_t *KernelAlternatives);

  void setGlobalOffset(const size_t *GlobalWorkOffsetPtr) {
    const size_t CopySize = sizeof(size_t) * WorkDim;
    std::memcpy(GlobalWorkOffset, GlobalWorkOffsetPtr, CopySize);
    if (WorkDim < 3) {
      const size_t ZeroSize = sizeof(size_t) * (3 - WorkDim);
      std::memset(GlobalWorkOffset + WorkDim, 0, ZeroSize);
    }
  }

  void setGlobalSize(const size_t *GlobalWorkSizePtr) {
    const size_t CopySize = sizeof(size_t) * WorkDim;
    std::memcpy(GlobalWorkSize, GlobalWorkSizePtr, CopySize);
    if (WorkDim < 3) {
      const size_t ZeroSize = sizeof(size_t) * (3 - WorkDim);
      std::memset(GlobalWorkSize + WorkDim, 0, ZeroSize);
    }
  }

  void setLocalSize(const size_t *LocalWorkSizePtr) {
    const size_t CopySize = sizeof(size_t) * WorkDim;
    std::memcpy(LocalWorkSize, LocalWorkSizePtr, CopySize);
    if (WorkDim < 3) {
      const size_t ZeroSize = sizeof(size_t) * (3 - WorkDim);
      std::memset(LocalWorkSize + WorkDim, 0, ZeroSize);
    }
  }

  void setNullLocalSize() noexcept {
    std::memset(LocalWorkSize, 0, sizeof(size_t) * 3);
  }

  bool isNullLocalSize() const noexcept {
    const size_t Zeros[3] = {0, 0, 0};
    return 0 == std::memcmp(LocalWorkSize, Zeros, sizeof(LocalWorkSize));
  }

  // The currently active kernel handle for this command.
  ur_kernel_handle_t Kernel;

  // Set of all the kernel handles that can be used when updating this command.
  std::unordered_set<ur_kernel_handle_t> ValidKernelHandles;

  CUDA_KERNEL_NODE_PARAMS Params;

  uint32_t WorkDim;
  size_t GlobalWorkOffset[3];
  size_t GlobalWorkSize[3];
  size_t LocalWorkSize[3];
};

struct fill_command_data {
  std::vector<CUgraphNode> DecomposedNodes;
};

// Command handle that can be returned from command append entry-points.
// The type of the command is specified by a CommandType field, with
// additional command-type-specific data stored in the CommandData enum.
struct ur_exp_command_buffer_command_handle_t_ : ur::cuda::handle_base {
  using command_data_type_t =
      std::variant<null_command_data, kernel_command_data, fill_command_data>;

  ur_exp_command_buffer_command_handle_t_(
      CommandType Type, ur_exp_command_buffer_handle_t CommandBuffer,
      CUgraphNode Node, CUgraphNode SignalNode,
      const std::vector<CUgraphNode> &WaitNodes,
      command_data_type_t Data = null_command_data{})
      : handle_base(), CommandBuffer(CommandBuffer), Node(Node),
        SignalNode(SignalNode), WaitNodes(WaitNodes), Type(Type),
        CommandData(Data) {}

  // Parent UR command-buffer.
  ur_exp_command_buffer_handle_t CommandBuffer;
  // Node created in graph for the command.
  CUgraphNode Node;
  // An optional EventRecordNode that's a successor of Node to signal
  // dependent commands outwith the command-buffer.
  CUgraphNode SignalNode;
  // Optional list of EventWait Nodes to wait on commands from outside of the
  // command-buffer.
  std::vector<CUgraphNode> WaitNodes;

  CommandType Type;
  command_data_type_t CommandData;
};

struct ur_exp_command_buffer_handle_t_ : ur::cuda::handle_base {

  ur_exp_command_buffer_handle_t_(ur_context_handle_t Context,
                                  ur_device_handle_t Device, bool IsUpdatable,
                                  bool IsInOrder);

  ~ur_exp_command_buffer_handle_t_();

  void registerSyncPoint(ur_exp_command_buffer_sync_point_t SyncPoint,
                         CUgraphNode CuNode) {
    SyncPoints[SyncPoint] = CuNode;
    NextSyncPoint++;
  }

  ur_exp_command_buffer_sync_point_t getNextSyncPoint() const {
    return NextSyncPoint;
  }

  // Creates a cuEvent object and adds a cuGraphAddEventRecordNode node to the
  // graph.
  // @param[in] DepNode Node for the EventRecord node to depend on.
  // @param[out] SignalNode Node created by cuGraphAddEventRecordNode.
  // @return UR event backed by CuEvent object that will be recorded to.
  std::unique_ptr<ur_event_handle_t_> addSignalNode(CUgraphNode DepNode,
                                                    CUgraphNode &SignalNode);

  // Adds a cuGraphAddEventWaitNodes node to the graph
  // @param[in,out] Dependencies for each of the wait nodes created. Set to the
  // list of wait nodes created on success.
  // @param[in] NumEventsInWaitList Number of wait nodes to create.
  // @param[in] UR events wrapping the cuEvent objects the nodes will wait on.
  // @returns UR_RESULT_SUCCESS or an error
  ur_result_t addWaitNodes(std::vector<CUgraphNode> &DepsList,
                           uint32_t NumEventsInWaitList,
                           const ur_event_handle_t *EventWaitList);

  // Helper to register next sync point
  // @param CuNode Node to register as next sync point
  // @return Pointer to the sync that registers the Node
  ur_exp_command_buffer_sync_point_t addSyncPoint(CUgraphNode CuNode) {
    ur_exp_command_buffer_sync_point_t SyncPoint = NextSyncPoint;
    registerSyncPoint(SyncPoint, std::move(CuNode));
    return SyncPoint;
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }
  uint32_t decrementReferenceCount() noexcept { return --RefCount; }
  uint32_t getReferenceCount() const noexcept { return RefCount; }

  // UR context associated with this command-buffer
  ur_context_handle_t Context;
  // Device associated with this command-buffer
  ur_device_handle_t Device;
  // Whether commands in the command-buffer can be updated
  bool IsUpdatable;
  // Whether commands in the command-buffer are in-order.
  bool IsInOrder;
  // Cuda Graph handle
  CUgraph CudaGraph;
  // Cuda Graph Exec handle
  CUgraphExec CudaGraphExec = nullptr;
  // Atomic variable counting the number of reference to this command_buffer
  // using std::atomic prevents data race when incrementing/decrementing.
  std::atomic_uint32_t RefCount;

  // Ordered map of sync_points to ur_events, so that we can find the last
  // node added to an in-order command-buffer.
  std::map<ur_exp_command_buffer_sync_point_t, CUgraphNode> SyncPoints;
  // Next sync_point value (may need to consider ways to reuse values if 32-bits
  // is not enough)
  ur_exp_command_buffer_sync_point_t NextSyncPoint;

  // Handles to individual commands in the command-buffer
  std::vector<std::unique_ptr<ur_exp_command_buffer_command_handle_t_>>
      CommandHandles;
};
