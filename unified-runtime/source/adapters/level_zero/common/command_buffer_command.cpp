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
#include "ur_ddi.h"
#include <ur/ur.hpp>

namespace {
// Dispatch kernel retain/release through the handle's ddi_table. Both v1 and v2
// kernel types derive from ur::handle_base<> whose first member is ddi_table,
// so a reinterpret_cast to handle_base_no_ddi lets us access it without knowing
// the full adapter-specific type.
inline const ur_dditable_t *getDdi(ur_kernel_handle_t kernel) {
  return reinterpret_cast<ur::handle_base_no_ddi *>(kernel)->ddi_table;
}
} // namespace

kernel_command_handle::kernel_command_handle(
    ur_exp_command_buffer_handle_t commandBuffer, ur_kernel_handle_t kernel,
    uint64_t commandId, uint32_t workDim, uint32_t numKernelAlternatives,
    ur_kernel_handle_t *kernelAlternatives)
    : ur_exp_command_buffer_command_handle_t_(commandBuffer, commandId),
      workDim(workDim), kernel(kernel) {
  // Add the default kernel to the list of valid kernels
  getDdi(kernel)->Kernel.pfnRetain(kernel);
  validKernelHandles.insert(kernel);
  // Add alternative kernels if provided
  if (kernelAlternatives) {
    for (size_t i = 0; i < numKernelAlternatives; i++) {
      getDdi(kernelAlternatives[i])->Kernel.pfnRetain(kernelAlternatives[i]);
      validKernelHandles.insert(kernelAlternatives[i]);
    }
  }
}

kernel_command_handle::~kernel_command_handle() {
  for (const ur_kernel_handle_t &kernelHandle : validKernelHandles) {
    getDdi(kernelHandle)->Kernel.pfnRelease(kernelHandle);
  }
}
