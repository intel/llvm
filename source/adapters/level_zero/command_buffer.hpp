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
#include "queue.hpp"

struct command_buffer_profiling_t {
  ur_exp_command_buffer_sync_point_t NumEvents;
  ze_kernel_timestamp_result_t *Timestamps;
};

struct ur_exp_command_buffer_handle_t_ : public _ur_object {
  ur_exp_command_buffer_handle_t_(ur_context_handle_t Context,
                                  ur_device_handle_t Device,
                                  ze_command_list_handle_t CommandList,
                                  ZeStruct<ze_command_list_desc_t> ZeDesc,
                                  const ur_exp_command_buffer_desc_t *Desc);

  ~ur_exp_command_buffer_handle_t_();

  void RegisterSyncPoint(ur_exp_command_buffer_sync_point_t SyncPoint,
                         ur_event_handle_t Event) {
    SyncPoints[SyncPoint] = Event;
    NextSyncPoint++;
  }

  ur_exp_command_buffer_sync_point_t GetNextSyncPoint() const {
    return NextSyncPoint;
  }

  // UR context associated with this command-buffer
  ur_context_handle_t Context;
  // Device associated with this command buffer
  ur_device_handle_t Device;
  // Level Zero command list handle
  ze_command_list_handle_t ZeCommandList;
  // Level Zero command list descriptor
  ZeStruct<ze_command_list_desc_t> ZeCommandListDesc;
  // Queue properties from command-buffer descriptor
  // TODO: Do we need these?
  ur_queue_properties_t QueueProperties;
  // Map of sync_points to ur_events
  std::unordered_map<ur_exp_command_buffer_sync_point_t, ur_event_handle_t>
      SyncPoints;
  // Next sync_point value (may need to consider ways to reuse values if 32-bits
  // is not enough)
  ur_exp_command_buffer_sync_point_t NextSyncPoint;
  // Command list map so we can use queue::executeCommandList.
  // Command list map is also used to release all the Fences retained by the
  // command_buffer std::unordered_multimap<ze_command_list_handle_t,
  // ur_command_list_info_t> CommandListMap; CommandListMap is redefined as a
  // multimap to enable mutiple commands enqueing into the same command_buffer
  std::unordered_multimap<ze_command_list_handle_t, ur_command_list_info_t>
      CommandListMap;
  // Event which will signals the most recent execution of the command-buffer
  // has finished
  ur_event_handle_t SignalEvent = nullptr;
  // Event which a command-buffer waits on until the wait-list dependencies
  // passed to a command-buffer enqueue have been satisfied.
  ur_event_handle_t WaitEvent = nullptr;
};
