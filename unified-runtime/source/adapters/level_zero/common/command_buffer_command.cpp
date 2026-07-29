//===--------- command_buffer_command.cpp - Level Zero Adapter ------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer_command.hpp"
#include "interfaces.hpp"

#include "unified-runtime/ur_api.h"
#include "unified-runtime/ur_ddi.h"

namespace ur::level_zero {

kernel_command_handle::kernel_command_handle(
    ur_exp_command_buffer_handle_t commandBuffer, ur_kernel_handle_t kernel,
    uint64_t commandId, uint32_t workDim, uint32_t numKernelAlternatives,
    ur_kernel_handle_t *kernelAlternatives)
    : ur_exp_command_buffer_command_handle_t_(commandBuffer, commandId),
      workDim(workDim), kernel(kernel) {
  // Add the default kernel to the list of valid kernels
  ddiTableOf(kernel)->Kernel.pfnRetain(kernel);
  validKernelHandles.insert(kernel);
  // Add alternative kernels if provided
  if (kernelAlternatives) {
    for (size_t i = 0; i < numKernelAlternatives; i++) {
      ddiTableOf(kernelAlternatives[i])
          ->Kernel.pfnRetain(kernelAlternatives[i]);
      validKernelHandles.insert(kernelAlternatives[i]);
    }
  }
}

kernel_command_handle::~kernel_command_handle() {
  for (const ur_kernel_handle_t &kernelHandle : validKernelHandles) {
    ddiTableOf(kernelHandle)->Kernel.pfnRelease(kernelHandle);
  }
}

} // namespace ur::level_zero
