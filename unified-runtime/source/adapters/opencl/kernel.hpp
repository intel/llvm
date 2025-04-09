//===--------- kernel.hpp - OpenCL Adapter ---------------------------===//
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
#include "context.hpp"
#include "program.hpp"

#include <vector>

struct ur_kernel_handle_t_ {
  using native_type = cl_kernel;
  native_type CLKernel;
  ur_program_handle_t Program;
  ur_context_handle_t Context;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = true;

  ur_kernel_handle_t_(native_type Kernel, ur_program_handle_t Program,
                      ur_context_handle_t Context)
      : CLKernel(Kernel), Program(Program), Context(Context) {
    RefCount = 1;
    urProgramRetain(Program);
    urContextRetain(Context);
  }

  ~ur_kernel_handle_t_() {
    urProgramRelease(Program);
    urContextRelease(Context);
    if (IsNativeHandleOwned) {
      clReleaseKernel(CLKernel);
    }
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  static ur_result_t makeWithNative(native_type NativeKernel,
                                    ur_program_handle_t Program,
                                    ur_context_handle_t Context,
                                    ur_kernel_handle_t &Kernel);
};
