//===--------- program.hpp - OpenCL Adapter ---------------------------===//
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

#include <vector>

struct ur_program_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_program;
  native_type CLProgram;
  ur_context_handle_t Context;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = true;
  uint32_t NumDevices = 0;
  std::vector<ur_device_handle_t> Devices;

  ur_program_handle_t_(native_type Prog, ur_context_handle_t Ctx,
                       uint32_t NumDevices, ur_device_handle_t *Devs)
      : handle_base(), CLProgram(Prog), Context(Ctx), NumDevices(NumDevices) {
    RefCount = 1;
    urContextRetain(Context);
    for (uint32_t i = 0; i < NumDevices; i++) {
      Devices.push_back(Devs[i]);
    }
  }

  ~ur_program_handle_t_() {
    urContextRelease(Context);
    if (IsNativeHandleOwned) {
      clReleaseProgram(CLProgram);
    }
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

  static ur_result_t makeWithNative(native_type NativeProg,
                                    ur_context_handle_t Context,
                                    ur_program_handle_t &Program);
};
