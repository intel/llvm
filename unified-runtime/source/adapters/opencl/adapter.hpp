//===-------------- adapter.hpp - OpenCL Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/cl.h"
#include "logger/ur_logger.hpp"

struct ur_adapter_handle_t_ {
  ur_adapter_handle_t_();

  std::atomic<uint32_t> RefCount = 0;
  std::mutex Mutex;
  logger::Logger &log = logger::get_logger("opencl");

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
