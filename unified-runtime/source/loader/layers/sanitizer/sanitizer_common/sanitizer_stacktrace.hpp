/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer_stacktrace.hpp
 *
 */

#pragma once

#include "sanitizer_common.hpp"

#include <vector>

namespace ur_sanitizer_layer {

constexpr size_t MAX_BACKTRACE_FRAMES = 64;

struct StackTrace {
  std::vector<BacktraceFrame> stack;

  void print() const;
};

StackTrace GetCurrentBacktrace();

char **GetBacktraceSymbols(const std::vector<BacktraceFrame> &BacktraceFrames);

} // namespace ur_sanitizer_layer
