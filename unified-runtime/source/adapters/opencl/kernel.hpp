//===--------- kernel.hpp - OpenCL Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "adapter.hpp"
#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "context.hpp"
#include "program.hpp"

#include <vector>

namespace ur::opencl {

struct ur_kernel_handle_t_ : handle_base {
  using native_type = cl_kernel;
  native_type CLKernel;
  ur_program_handle_t_ *Program;
  ur_context_handle_t_ *Context;
  bool IsNativeHandleOwned = true;
  clSetKernelArgMemPointerINTEL_fn clSetKernelArgMemPointerINTEL = nullptr;
  ur::RefCount RefCount;

  ur_kernel_handle_t_(const ur_kernel_handle_t_ &) = delete;
  ur_kernel_handle_t_ &operator=(const ur_kernel_handle_t_ &) = delete;

  ur_kernel_handle_t_(native_type Kernel, ur_program_handle_t_ *Program,
                      ur_context_handle_t_ *Context)
      : handle_base(), CLKernel(Kernel), Program(Program), Context(Context) {
    ur::opencl::urProgramRetain(ur_cast<ur_program_handle_t>(Program));
    ur::opencl::urContextRetain(ur_cast<ur_context_handle_t>(Context));

    cl_ext::getExtFuncFromContext<clSetKernelArgMemPointerINTEL_fn>(
        Context->CLContext,
        ur_cast<ur_adapter_handle_t_ *>(ur::cl::getAdapter())
            ->fnCache.clSetKernelArgMemPointerINTELCache,
        cl_ext::SetKernelArgMemPointerName, &clSetKernelArgMemPointerINTEL);
  }

  ~ur_kernel_handle_t_() {
    ur::opencl::urProgramRelease(ur_cast<ur_program_handle_t>(Program));
    ur::opencl::urContextRelease(ur_cast<ur_context_handle_t>(Context));
    if (IsNativeHandleOwned) {
      clReleaseKernel(CLKernel);
    }
  }

  static ur_result_t makeWithNative(native_type NativeKernel,
                                    ur_program_handle_t_ *Program,
                                    ur_context_handle_t_ *Context,
                                    ur_kernel_handle_t_ *&Kernel);
};

} // namespace ur::opencl
