//===--------- program.hpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

#include <atomic>

#include "context.hpp"

/// Implementation of UR Program on HIP Module object
struct ur_program_handle_t_ {
  using native_type = hipModule_t;
  native_type Module;
  const char *Binary;
  size_t BinarySizeInBytes;
  std::atomic_uint32_t RefCount;
  ur_context_handle_t Context;

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char ErrorLog[MAX_LOG_SIZE], InfoLog[MAX_LOG_SIZE];
  std::string BuildOptions;
  ur_program_build_status_t BuildStatus = UR_PROGRAM_BUILD_STATUS_NONE;

  ur_program_handle_t_(ur_context_handle_t Ctxt);
  ~ur_program_handle_t_();

  ur_result_t setBinary(const char *Binary, size_t BinarySizeInBytes);

  ur_result_t buildProgram(const char *BuildOptions);
  ur_context_handle_t getContext() const { return Context; };

  native_type get() const noexcept { return Module; };

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }
};
