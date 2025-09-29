//===--------- sampler.hpp - OpenCL Adapter ---------------------------===//
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

#include <vector>

struct ur_sampler_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_sampler;
  native_type CLSampler;
  ur_context_handle_t Context;
  bool IsNativeHandleOwned = false;
  ur::RefCount RefCount;

  ur_sampler_handle_t_(const ur_sampler_handle_t_ &) = delete;
  ur_sampler_handle_t_ &operator=(const ur_sampler_handle_t_ &) = delete;

  ur_sampler_handle_t_(native_type Sampler, ur_context_handle_t Ctx)
      : handle_base(), CLSampler(Sampler), Context(Ctx) {
    urContextRetain(Context);
  }

  ~ur_sampler_handle_t_() {
    urContextRelease(Context);
    if (IsNativeHandleOwned) {
      clReleaseSampler(CLSampler);
    }
  }
};
