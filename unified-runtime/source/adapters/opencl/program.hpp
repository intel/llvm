//===--------- program.hpp - OpenCL Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "context.hpp"

#include <vector>

namespace ur::opencl {

struct ur_program_handle_t_ : handle_base {
  using native_type = cl_program;
  native_type CLProgram;
  ur_context_handle_t_ *Context;
  bool IsNativeHandleOwned = true;
  uint32_t NumDevices = 0;
  std::vector<ur_device_handle_t_ *> Devices;
  ur::RefCount RefCount;

  ur_program_handle_t_(native_type Prog, ur_context_handle_t_ *Ctx,
                       uint32_t NumDevices, ur_device_handle_t_ **Devs)
      : handle_base(), CLProgram(Prog), Context(Ctx), NumDevices(NumDevices) {
    ur::opencl::urContextRetain(cast(Context));
    for (uint32_t i = 0; i < NumDevices; i++) {
      Devices.push_back(Devs[i]);
    }
  }

  ~ur_program_handle_t_() {
    ur::opencl::urContextRelease(cast(Context));
    if (IsNativeHandleOwned) {
      clReleaseProgram(CLProgram);
    }
  }

  static ur_result_t makeWithNative(native_type NativeProg,
                                    ur_context_handle_t Context,
                                    ur_program_handle_t &Program);
};

} // namespace ur::opencl
