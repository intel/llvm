/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include "backtrace.hpp"
#include <execinfo.h>

namespace ur {

std::vector<BacktraceLine> getCurrentBacktrace() {
  void *backtraceFrames[MAX_BACKTRACE_FRAMES];
  int frameCount = ::backtrace(backtraceFrames, MAX_BACKTRACE_FRAMES);
  char **backtraceStr = ::backtrace_symbols(backtraceFrames, frameCount);
  // TODO: implement getting demangled symbols using abi::__cxa_demangle
  if (backtraceStr == nullptr) {
    return std::vector<BacktraceLine>(1, "Failed to acquire a backtrace");
  }

  std::vector<BacktraceLine> backtrace;
  try {
    for (int i = 0; i < frameCount; i++) {
      backtrace.emplace_back(backtraceStr[i]);
    }
  } catch (std::bad_alloc &) {
    free(backtraceStr);
    return std::vector<BacktraceLine>(1, "Failed to acquire a backtrace");
  }

  free(backtraceStr);

  return backtrace;
}

} // namespace ur
