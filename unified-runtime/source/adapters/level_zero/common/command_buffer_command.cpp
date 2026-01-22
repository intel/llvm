//===--------- command_buffer_command.cpp - Level Zero Adapter ------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer_command.hpp"
#include "context_interface.hpp"

#include "unified-runtime/ur_api.h"
#include "unified-runtime/ur_ddi.h"

namespace ur::level_zero::common {

namespace {
// Dispatch kernel retain/release through the handle's ddi_table. Every UR
// handle carries ddi_table at offset 0 (loader intercept requires this), so
// `ddiTableOf()` reads it without needing the concrete struct definition.
inline const ur_dditable_t *getDdi(ur_kernel_handle_t kernel) {
  return ddiTableOf(kernel);
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

} // namespace ur::level_zero::common
