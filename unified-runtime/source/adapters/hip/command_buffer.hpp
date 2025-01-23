//===--------- command_buffer.hpp - HIP Adapter ---------------------------===//
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
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>

// Handle to a kernel command.
//
// Struct that stores all the information related to a kernel command in a
// command-buffer, such that the command can be recreated. When handles can
// be returned from other command types this struct will need refactored.
struct ur_exp_command_buffer_command_handle_t_ : ur::hip::handle_base {
  ur_exp_command_buffer_command_handle_t_(
      ur_exp_command_buffer_handle_t CommandBuffer, ur_kernel_handle_t Kernel,
      hipGraphNode_t Node, hipKernelNodeParams Params, uint32_t WorkDim,
      const size_t *GlobalWorkOffsetPtr, const size_t *GlobalWorkSizePtr,
      const size_t *LocalWorkSizePtr, uint32_t NumKernelAlternatives,
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

  ur_exp_command_buffer_handle_t CommandBuffer;

  // The currently active kernel handle for this command.
  ur_kernel_handle_t Kernel;

  // Set of all the kernel handles that can be used when updating this command.
  std::unordered_set<ur_kernel_handle_t> ValidKernelHandles;

  hipGraphNode_t Node;
  hipKernelNodeParams Params;

  uint32_t WorkDim;
  size_t GlobalWorkOffset[3];
  size_t GlobalWorkSize[3];
  size_t LocalWorkSize[3];
};

struct ur_exp_command_buffer_handle_t_ : ur::hip::handle_base {

  ur_exp_command_buffer_handle_t_(ur_context_handle_t hContext,
                                  ur_device_handle_t hDevice, bool IsUpdatable);

  ~ur_exp_command_buffer_handle_t_();

  void registerSyncPoint(ur_exp_command_buffer_sync_point_t SyncPoint,
                         hipGraphNode_t HIPNode) {
    SyncPoints[SyncPoint] = std::move(HIPNode);
    NextSyncPoint++;
  }

  ur_exp_command_buffer_sync_point_t getNextSyncPoint() const {
    return NextSyncPoint;
  }

  // Helper to register next sync point
  // @param HIPNode Node to register as next sync point
  // @return Pointer to the sync that registers the Node
  ur_exp_command_buffer_sync_point_t addSyncPoint(hipGraphNode_t HIPNode) {
    ur_exp_command_buffer_sync_point_t SyncPoint = NextSyncPoint;
    registerSyncPoint(SyncPoint, std::move(HIPNode));
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
  // HIP Graph handle
  hipGraph_t HIPGraph;
  // HIP Graph Exec handle
  hipGraphExec_t HIPGraphExec = nullptr;
  // Atomic variable counting the number of reference to this command_buffer
  // using std::atomic prevents data race when incrementing/decrementing.
  std::atomic_uint32_t RefCount;

  // Map of sync_points to ur_events
  std::unordered_map<ur_exp_command_buffer_sync_point_t, hipGraphNode_t>
      SyncPoints;
  // Next sync_point value (may need to consider ways to reuse values if 32-bits
  // is not enough)
  ur_exp_command_buffer_sync_point_t NextSyncPoint;

  // Handles to individual commands in the command-buffer
  std::vector<std::unique_ptr<ur_exp_command_buffer_command_handle_t_>>
      CommandHandles;
};
