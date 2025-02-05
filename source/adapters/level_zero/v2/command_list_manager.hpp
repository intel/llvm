//===--------- command_list_manager.hpp - Level Zero Adapter --------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "command_list_cache.hpp"
#include "common.hpp"
#include "event_pool_cache.hpp"
#include "queue_api.hpp"
#include <ze_api.h>

struct wait_list_view {
  ze_event_handle_t *handles;
  uint32_t num;

  operator bool() const {
    assert((handles != nullptr) == (num > 0));
    return handles != nullptr;
  }

  void clear() {
    handles = nullptr;
    num = 0;
  }
};

struct ur_command_list_manager : public _ur_object {

  ur_command_list_manager(ur_context_handle_t context,
                          ur_device_handle_t device,
                          v2::raii::command_list_unique_handle &&commandList,
                          v2::event_flags_t flags = v2::EVENT_FLAGS_COUNTER,
                          ur_queue_t_ *queue = nullptr);
  ~ur_command_list_manager();

  ur_result_t appendKernelLaunch(ur_kernel_handle_t hKernel, uint32_t workDim,
                                 const size_t *pGlobalWorkOffset,
                                 const size_t *pGlobalWorkSize,
                                 const size_t *pLocalWorkSize,
                                 uint32_t numEventsInWaitList,
                                 const ur_event_handle_t *phEventWaitList,
                                 ur_event_handle_t *phEvent);

  ze_command_list_handle_t getZeCommandList();

  wait_list_view getWaitListView(const ur_event_handle_t *phWaitEvents,
                                 uint32_t numWaitEvents);
  ze_event_handle_t getSignalEvent(ur_event_handle_t *hUserEvent,
                                   ur_command_t commandType);

private:
  // UR context associated with this command-buffer
  ur_context_handle_t context;
  // Device associated with this command-buffer
  ur_device_handle_t device;
  v2::raii::cache_borrowed_event_pool eventPool;
  v2::raii::command_list_unique_handle zeCommandList;
  ur_queue_t_ *queue;
  std::vector<ze_event_handle_t> waitList;
};
