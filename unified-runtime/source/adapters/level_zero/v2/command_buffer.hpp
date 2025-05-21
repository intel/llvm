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

#include "../helpers/mutable_helpers.hpp"
#include "command_list_manager.hpp"
#include "common.hpp"
#include "context.hpp"
#include "kernel.hpp"
#include "lockable.hpp"
#include "queue_api.hpp"
#include <unordered_set>
#include <ze_api.h>
struct kernel_command_handle;

struct ur_exp_command_buffer_handle_t_ : public ur_object {
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

  ur_result_t
  createCommandHandle(locked<ur_command_list_manager> &commandListLocked,
                      ur_kernel_handle_t hKernel, uint32_t workDim,
                      const size_t *pGlobalWorkSize,
                      uint32_t numKernelAlternatives,
                      ur_kernel_handle_t *kernelAlternatives,
                      ur_exp_command_buffer_command_handle_t *command);
  ur_result_t applyUpdateCommands(
      uint32_t numUpdateCommands,
      const ur_exp_command_buffer_update_kernel_launch_desc_t *updateCommands);

private:
  const ur_context_handle_t context;
  const ur_device_handle_t device;
  std::vector<std::unique_ptr<ur_exp_command_buffer_command_handle_t_>>
      commandHandles;
  // Indicates if command-buffer was finalized.
  bool isFinalized = false;

  ur_event_handle_t currentExecution = nullptr;
};
