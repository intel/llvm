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

struct ur_program_handle_t_ : _ur_object {
  using native_type = hipModule_t;
  native_type module_;
  const char *binary_;
  size_t binarySizeInBytes_;
  std::atomic_uint32_t refCount_;
  ur_context_handle_t context_;

  constexpr static size_t MAX_LOG_SIZE = 8192u;

  char errorLog_[MAX_LOG_SIZE], infoLog_[MAX_LOG_SIZE];
  std::string buildOptions_;
  ur_program_build_status_t buildStatus_ = UR_PROGRAM_BUILD_STATUS_NONE;

  ur_program_handle_t_(ur_context_handle_t ctxt);
  ~ur_program_handle_t_();

  ur_result_t set_binary(const char *binary, size_t binarySizeInBytes);

  ur_result_t build_program(const char *build_options);
  ur_context_handle_t get_context() const { return context_; };

  native_type get() const noexcept { return module_; };

  uint32_t increment_reference_count() noexcept { return ++refCount_; }

  uint32_t decrement_reference_count() noexcept { return --refCount_; }

  uint32_t get_reference_count() const noexcept { return refCount_; }
};
