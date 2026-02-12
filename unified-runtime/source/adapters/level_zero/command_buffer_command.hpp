//===--------- command_buffer_command.hpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "common.hpp"
#include <unordered_set>
#include <ur_api.h>

struct ur_exp_command_buffer_command_handle_t_ : public ur_object {
  ur_exp_command_buffer_command_handle_t_(
      ur_exp_command_buffer_handle_t commandBuffer, uint64_t commandId)
      : commandBuffer(commandBuffer), commandId(commandId) {}

  virtual ~ur_exp_command_buffer_command_handle_t_() {}

  // Command-buffer of this command.
  ur_exp_command_buffer_handle_t commandBuffer;
  // L0 command ID identifying this command
  uint64_t commandId;
};

struct kernel_command_handle : public ur_exp_command_buffer_command_handle_t_ {
  kernel_command_handle(ur_exp_command_buffer_handle_t commandBuffer,
                        ur_kernel_handle_t kernel, uint64_t commandId,
                        uint32_t workDim, uint32_t numKernelAlternatives,
                        ur_kernel_handle_t *kernelAlternatives);

  ~kernel_command_handle();

  void setGlobalWorkSize(const size_t *globalWorkSizePtr) {
    const size_t copySize = sizeof(size_t) * workDim;
    std::memcpy(globalWorkSize, globalWorkSizePtr, copySize);
    if (workDim < 3) {
      const size_t zeroSize = sizeof(size_t) * (3 - workDim);
      std::memset(globalWorkSize + workDim, 0, zeroSize);
    }
  }

  // Work-dimension the command was originally created with.
  uint32_t workDim;
  // Global work size of the kernel
  size_t globalWorkSize[3];
  // Currently active kernel handle
  ur_kernel_handle_t kernel;
  // Storage for valid kernel alternatives for this command.
  std::unordered_set<ur_kernel_handle_t> validKernelHandles;
};
