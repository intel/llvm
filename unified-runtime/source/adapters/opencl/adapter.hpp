//===-------------- adapter.hpp - OpenCL Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "device.hpp"
#include "logger/ur_logger.hpp"
#include "platform.hpp"

#include "CL/cl.h"
#include "common.hpp"
#include "logger/ur_logger.hpp"

struct ur_adapter_handle_t_ : ur::opencl::handle_base {
  ur_adapter_handle_t_();
  ~ur_adapter_handle_t_();

  ur_adapter_handle_t_(ur_adapter_handle_t_ &) = delete;

  std::atomic<uint32_t> RefCount = 0;
  logger::Logger &log = logger::get_logger("opencl");
  cl_ext::ExtFuncPtrCacheT fnCache{};

  std::vector<std::unique_ptr<ur_platform_handle_t_>> URPlatforms;
  uint32_t NumPlatforms = 0;

  // Function pointers to core OpenCL entry points which may not exist in older
  // versions of the OpenCL-ICD-Loader are tracked here and initialized by
  // dynamically loading the symbol by name.
#define CL_CORE_FUNCTION(FUNC) decltype(::FUNC) *FUNC = nullptr;
#include "core_functions.def"
#undef CL_CORE_FUNCTION
};

namespace ur {
namespace cl {
ur_adapter_handle_t getAdapter();
} // namespace cl
} // namespace ur
