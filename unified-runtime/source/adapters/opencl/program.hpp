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
#include "common/ur_ref_count.hpp"
#include "context.hpp"

#include <vector>

struct ur_program_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_program;
  native_type CLProgram;
  ur_context_handle_t Context;
  bool IsNativeHandleOwned = true;
  uint32_t NumDevices = 0;
  std::vector<ur_device_handle_t> Devices;
  ur::RefCount RefCount;

  ur_program_handle_t_(native_type Prog, ur_context_handle_t Ctx,
                       uint32_t NumDevices, ur_device_handle_t *Devs)
      : handle_base(), CLProgram(Prog), Context(Ctx), NumDevices(NumDevices) {
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

  static ur_result_t makeWithNative(native_type NativeProg,
                                    ur_context_handle_t Context,
                                    ur_program_handle_t &Program);
};
