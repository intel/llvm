//===-------------- adapter.hpp - OpenCL Adapter ---------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "device.hpp"
#include "logger/ur_logger.hpp"
#include "platform.hpp"

#include "CL/cl.h"
#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "logger/ur_logger.hpp"

struct ur_adapter_handle_t_ : ur::opencl::handle_base {
  ur_adapter_handle_t_();
  ~ur_adapter_handle_t_();

  ur_adapter_handle_t_(ur_adapter_handle_t_ &) = delete;
  ur_adapter_handle_t_ &operator=(const ur_adapter_handle_t_ &) = delete;

  logger::Logger &log = logger::get_logger("opencl");
  cl_ext::ExtFuncPtrCacheT fnCache{};

  std::vector<std::unique_ptr<ur_platform_handle_t_>> URPlatforms;
  uint32_t NumPlatforms = 0;

  ur::RefCount RefCount;

  // Function pointers to core OpenCL entry points which may not exist in older
  // versions of the OpenCL-ICD-Loader are tracked here and initialized by
  // dynamically loading the symbol by name.
#ifdef UR_STATIC_ADAPTER_OPENCL
  // Lift redirect macros so decltype resolves the real CL signatures
#undef clSetProgramSpecializationConstant
#undef clSetContextDestructorCallback
#endif
#define CL_CORE_FUNCTION(FUNC) decltype(::FUNC) *FUNC = nullptr;
#include "core_functions.def"
#undef CL_CORE_FUNCTION
#ifdef UR_STATIC_ADAPTER_OPENCL
  // Restore redirect macros so direct calls in this TU still go via pointers
#define clSetProgramSpecializationConstant                                     \
  ocl::clSetProgramSpecializationConstant_ptr
#define clSetContextDestructorCallback ocl::clSetContextDestructorCallback_ptr
#endif
};

namespace ur {
namespace cl {
ur_adapter_handle_t getAdapter();
} // namespace cl
} // namespace ur
