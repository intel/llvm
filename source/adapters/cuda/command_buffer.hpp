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

// Trace an internal UR call
#define UR_TRACE(Call)                                                         \
  {                                                                            \
    ur_result_t Result;                                                        \
    UR_CALL(Call, Result);                                                     \
  }

// Trace an internal UR call and return the result to the user.
#define UR_CALL(Call, Result)                                                  \
  {                                                                            \
    if (PrintTrace)                                                            \
      logger::always("UR ---> {}", #Call);                                     \
    Result = (Call);                                                           \
    if (PrintTrace)                                                            \
      logger::always("UR <--- {}({})", #Call, Result);                         \
  }

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

// Command handle that can be returned from command append entry-points.
// Implemented as an abstract base class that handles for the specific
// command types derive from.
struct ur_exp_command_buffer_command_handle_t_ {
  ur_exp_command_buffer_command_handle_t_(
      ur_exp_command_buffer_handle_t CommandBuffer, CUgraphNode Node,
      CUgraphNode SignalNode, const std::vector<CUgraphNode> &WaitNodes);

  virtual ~ur_exp_command_buffer_command_handle_t_() {}

  virtual CommandType getCommandType() const noexcept = 0;

  uint32_t incrementInternalReferenceCount() noexcept {
    return ++RefCountInternal;
  }
  uint32_t decrementInternalReferenceCount() noexcept {
    return --RefCountInternal;
  }

  uint32_t incrementExternalReferenceCount() noexcept {
    return ++RefCountExternal;
  }
  uint32_t decrementExternalReferenceCount() noexcept {
    return --RefCountExternal;
  }
  uint32_t getExternalReferenceCount() const noexcept {
    return RefCountExternal;
  }

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

private:
  std::atomic_uint32_t RefCountInternal;
  std::atomic_uint32_t RefCountExternal;
};

struct kernel_command_handle : ur_exp_command_buffer_command_handle_t_ {
  kernel_command_handle(
      ur_exp_command_buffer_handle_t CommandBuffer, ur_kernel_handle_t Kernel,
      CUgraphNode Node, CUDA_KERNEL_NODE_PARAMS Params, uint32_t WorkDim,
      const size_t *GlobalWorkOffsetPtr, const size_t *GlobalWorkSizePtr,
      const size_t *LocalWorkSizePtr, uint32_t NumKernelAlternatives,
      ur_kernel_handle_t *KernelAlternatives, CUgraphNode SignalNode,
      const std::vector<CUgraphNode> &WaitNodes);

  CommandType getCommandType() const noexcept override {
    return CommandType::Kernel;
  }

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

struct usm_memcpy_command_handle : ur_exp_command_buffer_command_handle_t_ {
  usm_memcpy_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                            CUgraphNode Node, CUgraphNode SignalNode,
                            const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::USMMemcpy;
  }
};

struct usm_fill_command_handle : ur_exp_command_buffer_command_handle_t_ {
  usm_fill_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                          CUgraphNode Node, CUgraphNode SignalNode,
                          const std::vector<CUgraphNode> &WaitNodes,
                          const std::vector<CUgraphNode> &DecomposedNodes = {})
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes),
        DecomposedNodes(std::move(DecomposedNodes)) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::USMFill;
  }

  // If this fill command was decomposed into multiple nodes, this vector
  // contains all of those nodes in the order they were added to the graph.
  // Currently unused but will be required for updating in future.
  std::vector<CUgraphNode> DecomposedNodes;
};

struct buffer_copy_command_handle : ur_exp_command_buffer_command_handle_t_ {
  buffer_copy_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                             CUgraphNode Node, CUgraphNode SignalNode,
                             const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::MemBufferCopy;
  }
};

struct buffer_copy_rect_command_handle
    : ur_exp_command_buffer_command_handle_t_ {
  buffer_copy_rect_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                                  CUgraphNode Node, CUgraphNode SignalNode,
                                  const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::MemBufferCopyRect;
  }
};

struct buffer_read_command_handle : ur_exp_command_buffer_command_handle_t_ {
  buffer_read_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                             CUgraphNode Node, CUgraphNode SignalNode,
                             const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::MemBufferRead;
  }
};

struct buffer_read_rect_command_handle
    : ur_exp_command_buffer_command_handle_t_ {
  buffer_read_rect_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                                  CUgraphNode Node, CUgraphNode SignalNode,
                                  const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::MemBufferReadRect;
  }
};

struct buffer_write_command_handle : ur_exp_command_buffer_command_handle_t_ {
  buffer_write_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                              CUgraphNode Node, CUgraphNode SignalNode,
                              const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::MemBufferWrite;
  }
};

struct buffer_write_rect_command_handle
    : ur_exp_command_buffer_command_handle_t_ {
  buffer_write_rect_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                                   CUgraphNode Node, CUgraphNode SignalNode,
                                   const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::MemBufferWriteRect;
  }
};

struct buffer_fill_command_handle : ur_exp_command_buffer_command_handle_t_ {
  buffer_fill_command_handle(
      ur_exp_command_buffer_handle_t CommandBuffer, CUgraphNode Node,
      CUgraphNode SignalNode, const std::vector<CUgraphNode> &WaitNodes,
      const std::vector<CUgraphNode> &DecomposedNodes = {})
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes),
        DecomposedNodes(std::move(DecomposedNodes)) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::MemBufferFill;
  }

  // If this fill command was decomposed into multiple nodes, this vector
  // contains all of those nodes in the order they were added to the graph.
  // Currently unused but will be required for updating in future.
  std::vector<CUgraphNode> DecomposedNodes;
};

struct usm_prefetch_command_handle : ur_exp_command_buffer_command_handle_t_ {
  usm_prefetch_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                              CUgraphNode Node, CUgraphNode SignalNode,
                              const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::USMPrefetch;
  }
};

struct usm_advise_command_handle : ur_exp_command_buffer_command_handle_t_ {
  usm_advise_command_handle(ur_exp_command_buffer_handle_t CommandBuffer,
                            CUgraphNode Node, CUgraphNode SignalNode,
                            const std::vector<CUgraphNode> &WaitNodes)
      : ur_exp_command_buffer_command_handle_t_(CommandBuffer, Node, SignalNode,
                                                WaitNodes) {}
  CommandType getCommandType() const noexcept override {
    return CommandType::USMAdvise;
  }
};

struct ur_exp_command_buffer_handle_t_ {

  ur_exp_command_buffer_handle_t_(ur_context_handle_t Context,
                                  ur_device_handle_t Device, bool IsUpdatable);

  virtual ~ur_exp_command_buffer_handle_t_();

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

  uint32_t incrementInternalReferenceCount() noexcept {
    return ++RefCountInternal;
  }
  uint32_t decrementInternalReferenceCount() noexcept {
    return --RefCountInternal;
  }
  uint32_t getInternalReferenceCount() const noexcept {
    return RefCountInternal;
  }

  uint32_t incrementExternalReferenceCount() noexcept {
    return ++RefCountExternal;
  }
  uint32_t decrementExternalReferenceCount() noexcept {
    return --RefCountExternal;
  }
  uint32_t getExternalReferenceCount() const noexcept {
    return RefCountExternal;
  }

  // UR context associated with this command-buffer
  ur_context_handle_t Context;
  // Device associated with this command buffer
  ur_device_handle_t Device;
  // Whether commands in the command-buffer can be updated
  bool IsUpdatable;
  // Cuda Graph handle
  CUgraph CudaGraph;
  // Cuda Graph Exec handle
  CUgraphExec CudaGraphExec = nullptr;
  // Atomic variable counting the number of reference to this command_buffer
  // using std::atomic prevents data race when incrementing/decrementing.
  std::atomic_uint32_t RefCountInternal;
  std::atomic_uint32_t RefCountExternal;

  // Map of sync_points to ur_events
  std::unordered_map<ur_exp_command_buffer_sync_point_t, CUgraphNode>
      SyncPoints;
  // Next sync_point value (may need to consider ways to reuse values if 32-bits
  // is not enough)
  ur_exp_command_buffer_sync_point_t NextSyncPoint;

  // Handles to individual commands in the command-buffer
  std::vector<ur_exp_command_buffer_command_handle_t> CommandHandles;
};
