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

#include <vector>

struct ur_sampler_handle_t_ : ur::opencl::handle_base {
  using native_type = cl_sampler;
  native_type CLSampler;
  ur_context_handle_t Context;
  std::atomic<uint32_t> RefCount = 0;
  bool IsNativeHandleOwned = false;

  ur_sampler_handle_t_(native_type Sampler, ur_context_handle_t Ctx)
      : handle_base(), CLSampler(Sampler), Context(Ctx) {
    RefCount = 1;
    urContextRetain(Context);
  }

  ~ur_sampler_handle_t_() {
    urContextRelease(Context);
    if (IsNativeHandleOwned) {
      clReleaseSampler(CLSampler);
    }
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};
