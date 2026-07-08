//===--------- sampler.hpp - OpenCL Adapter ---------------------------===//
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

#include <vector>

namespace ur::opencl {

struct ur_context_handle_t_;

struct ur_sampler_handle_t_ : handle_base {
  using native_type = cl_sampler;
  native_type CLSampler;
  ur::opencl::ur_context_handle_t_ *Context;
  bool IsNativeHandleOwned = false;
  ur::RefCount RefCount;

  ur_sampler_handle_t_(const ur_sampler_handle_t_ &) = delete;
  ur_sampler_handle_t_ &operator=(const ur_sampler_handle_t_ &) = delete;

  ur_sampler_handle_t_(native_type Sampler,
                       ur::opencl::ur_context_handle_t_ *Ctx)
      : handle_base(), CLSampler(Sampler), Context(Ctx) {
    ur::opencl::urContextRetain(cast(Context));
  }

  ~ur_sampler_handle_t_() {
    ur::opencl::urContextRelease(cast(Context));
    if (IsNativeHandleOwned) {
      clReleaseSampler(CLSampler);
    }
  }
};

} // namespace ur::opencl
