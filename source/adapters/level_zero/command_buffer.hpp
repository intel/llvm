//===--------- command_buffer.hpp - Level Zero Adapter --------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "common.hpp"

#include "context.hpp"
#include "kernel.hpp"
#include "queue.hpp"

struct command_buffer_profiling_t {
  ur_exp_command_buffer_sync_point_t NumEvents;
  ze_kernel_timestamp_result_t *Timestamps;
};

struct ur_exp_command_buffer_handle_t_ : public _ur_object {
  ur_exp_command_buffer_handle_t_(
      ur_context_handle_t Context, ur_device_handle_t Device,
      ze_command_list_handle_t CommandList,
      ze_command_list_handle_t CommandListTranslated,
      ze_command_list_handle_t CommandListResetEvents,
      ze_command_list_handle_t CopyCommandList, ur_event_handle_t SignalEvent,
      ur_event_handle_t WaitEvent, ur_event_handle_t AllResetEvent,
      const ur_exp_command_buffer_desc_t *Desc, const bool IsInOrderCmdList);

  void registerSyncPoint(ur_exp_command_buffer_sync_point_t SyncPoint,
                         ur_event_handle_t Event);

  ur_exp_command_buffer_sync_point_t getNextSyncPoint() const {
    return NextSyncPoint;
  }

  // Indicates if a copy engine is available for use
  bool useCopyEngine() const { return ZeCopyCommandList != nullptr; }

  /**
   * Obtains a fence for a specific L0 queue. If there is already an available
   * fence for this queue, it will be reused.
   * @param[in] ZeCommandQueue The L0 queue associated with the fence.
   * @param[out] ZeFence The fence.
   * @return UR_RESULT_SUCCESS or an error code on failure
   */
  ur_result_t getFenceForQueue(ze_command_queue_handle_t &ZeCommandQueue,
                               ze_fence_handle_t &ZeFence);

  /**
   * Chooses which command list to use when appending a command to this command
   * buffer.
   * @param[in] PreferCopyEngine If true, will try to choose a copy engine
   * command-list. Will choose a compute command-list otherwise.
   * @return The chosen command list.
   */
  ze_command_list_handle_t chooseCommandList(bool PreferCopyEngine);

  // Releases the resources associated with the command-buffer before the
  // command-buffer object is destroyed.
  void cleanupCommandBufferResources();

  // UR context associated with this command-buffer
  ur_context_handle_t Context;
  // Device associated with this command buffer
  ur_device_handle_t Device;
  // Level Zero command list handle
  ze_command_list_handle_t ZeComputeCommandList;
  // Given a multi driver scenario, the driver handle must be translated to the
  // internal driver handle to allow calls to driver experimental apis.
  ze_command_list_handle_t ZeComputeCommandListTranslated;
  // Level Zero command list handle
  ze_command_list_handle_t ZeCommandListResetEvents;
  // Level Zero Copy command list handle
  ze_command_list_handle_t ZeCopyCommandList;
  // Event which will signals the most recent execution of the command-buffer
  // has finished
  ur_event_handle_t SignalEvent = nullptr;
  // Event which a command-buffer waits on until the wait-list dependencies
  // passed to a command-buffer enqueue have been satisfied.
  ur_event_handle_t WaitEvent = nullptr;
  // Event which a command-buffer waits on until the main command-list event
  // have been reset.
  ur_event_handle_t AllResetEvent = nullptr;
  // This flag is must be set to false if at least one copy command has been
  // added to `ZeCopyCommandList`
  bool MCopyCommandListEmpty = true;
  // Level Zero fences for each queue the command-buffer has been enqueued to.
  // These should be destroyed when the command-buffer is released.
  std::unordered_map<ze_command_queue_handle_t, ze_fence_handle_t> ZeFencesMap;
  // The Level Zero fence from the most recent enqueue of the command-buffer.
  // Must be an element in ZeFencesMap, so is not required to be destroyed
  // itself.
  ze_fence_handle_t ZeActiveFence;
  // Map of sync_points to ur_events
  std::unordered_map<ur_exp_command_buffer_sync_point_t, ur_event_handle_t>
      SyncPoints;
  // Next sync_point value (may need to consider ways to reuse values if 32-bits
  // is not enough)
  ur_exp_command_buffer_sync_point_t NextSyncPoint;
  // List of Level Zero events associated with submitted commands.
  std::vector<ze_event_handle_t> ZeEventsList;

  // Indicates if command-buffer commands can be updated after it is closed.
  bool IsUpdatable = false;
  // Indicates if command buffer was finalized.
  bool IsFinalized = false;
  // Command-buffer profiling is enabled.
  bool IsProfilingEnabled = false;
  // Command-buffer can be submitted to an in-order command-list.
  bool IsInOrderCmdList = false;
  // This list is needed to release all kernels retained by the
  // command_buffer.
  std::vector<ur_kernel_handle_t> KernelsList;
};

struct ur_exp_command_buffer_command_handle_t_ : public _ur_object {
  ur_exp_command_buffer_command_handle_t_(ur_exp_command_buffer_handle_t,
                                          uint64_t, uint32_t, bool,
                                          ur_kernel_handle_t);

  ~ur_exp_command_buffer_command_handle_t_();

  // Command-buffer of this command.
  ur_exp_command_buffer_handle_t CommandBuffer;

  uint64_t CommandId;
  // Work-dimension the command was originally created with.
  uint32_t WorkDim;
  // Set to true if the user set the local work size on command creation.
  bool UserDefinedLocalSize;
  ur_kernel_handle_t Kernel;
};
