//===--------- memory.hpp - OpenCL Adapter ---------------------------===//
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

struct ur_mem_handle_t_ : handle_base {
  using native_type = cl_mem;
  native_type CLMemory;
  ur_context_handle_t_ *Context;
  bool IsNativeHandleOwned = true;
  ur::RefCount RefCount;

  ur_mem_handle_t_(const ur_mem_handle_t_ &) = delete;
  ur_mem_handle_t_ &operator=(const ur_mem_handle_t_ &) = delete;

  ur_mem_handle_t_(native_type Mem, ur_context_handle_t_ *Ctx)
      : handle_base(), CLMemory(Mem), Context(Ctx) {
    ur::opencl::urContextRetain(cast(Context));
  }

  ~ur_mem_handle_t_() {
    ur::opencl::urContextRelease(cast(Context));
    if (IsNativeHandleOwned) {
      clReleaseMemObject(CLMemory);
    }
  }

  static ur_result_t makeWithNative(native_type NativeMem,
                                    ur_context_handle_t Ctx,
                                    ur_mem_handle_t &Mem);
};

} // namespace ur::opencl
