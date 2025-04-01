//===--------- command_buffer.hpp - Level Zero Adapter ---------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "command_list_manager.hpp"
#include "common.hpp"
#include "context.hpp"
#include "kernel.hpp"
#include "lockable.hpp"
#include "queue_api.hpp"
#include <ze_api.h>

struct ur_exp_command_buffer_handle_t_ : public _ur_object {
  ur_exp_command_buffer_handle_t_(
      ur_context_handle_t context, ur_device_handle_t device,
      v2::raii::command_list_unique_handle &&commandList,
      const ur_exp_command_buffer_desc_t *desc);

  ~ur_exp_command_buffer_handle_t_();

  ur_event_handle_t getExecutionEventUnlocked();
  ur_result_t
  registerExecutionEventUnlocked(ur_event_handle_t nextExecutionEvent);

  lockable<ur_command_list_manager> commandListManager;

  ur_result_t finalizeCommandBuffer();
  // Indicates if command-buffer commands can be updated after it is closed.
  const bool isUpdatable = false;
  // Command-buffer profiling is enabled.
  const bool isProfilingEnabled = false;

private:
  // Indicates if command-buffer was finalized.
  bool isFinalized = false;

  ur_event_handle_t currentExecution = nullptr;
};

struct ur_exp_command_buffer_command_handle_t_ : public _ur_object {
  ur_exp_command_buffer_command_handle_t_(ur_exp_command_buffer_handle_t,
                                          uint64_t);

private:
  ~ur_exp_command_buffer_command_handle_t_();

  // Command-buffer of this command.
  ur_exp_command_buffer_handle_t commandBuffer;
  // L0 command ID identifying this command
  uint64_t commandId;
};
