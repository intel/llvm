//===--------- command_buffer_command.cpp - Level Zero Adapter ------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer_command.hpp"

#include "ur_api.h"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"

kernel_command_handle::kernel_command_handle(
    ur_exp_command_buffer_handle_t commandBuffer, ur_kernel_handle_t kernel,
    uint64_t commandId, uint32_t workDim, uint32_t numKernelAlternatives,
    ur_kernel_handle_t *kernelAlternatives)
    : ur_exp_command_buffer_command_handle_t_(commandBuffer, commandId),
      workDim(workDim), kernel(kernel) {
  // Add the default kernel to the list of valid kernels
  ur::level_zero::urKernelRetain(kernel);
  validKernelHandles.insert(kernel);
  // Add alternative kernels if provided
  if (kernelAlternatives) {
    for (size_t i = 0; i < numKernelAlternatives; i++) {
      ur::level_zero::urKernelRetain(kernelAlternatives[i]);
      validKernelHandles.insert(kernelAlternatives[i]);
    }
  }
}

kernel_command_handle::~kernel_command_handle() {
  for (const ur_kernel_handle_t &kernelHandle : validKernelHandles) {
    ur::level_zero::urKernelRelease(kernelHandle);
  }
}
