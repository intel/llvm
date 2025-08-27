/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file backtrace.cpp
 *
 */
#include "sanitizer_common/sanitizer_stacktrace.hpp"

#include <execinfo.h>
#include <stdint.h>
#include <string>

namespace ur_sanitizer_layer {

StackTrace GetCurrentBacktrace() {
  BacktraceFrame Frames[MAX_BACKTRACE_FRAMES];
  int FrameCount = backtrace(Frames, MAX_BACKTRACE_FRAMES);

  // The Frames contain the return addresses, which is one instruction after the
  // call instruction. Adjust the addresses so that symbolizer would give more
  // precise result.
  for (int I = 0; I < FrameCount; I++) {
    Frames[I] = (void *)((uintptr_t)Frames[I] - 1);
  }

  StackTrace Stack;
  Stack.stack = std::vector<BacktraceFrame>(&Frames[0], &Frames[FrameCount]);

  return Stack;
}

char **GetBacktraceSymbols(const std::vector<BacktraceFrame> &BacktraceFrames) {
  assert(!BacktraceFrames.empty());

  char **BacktraceSymbols =
      backtrace_symbols(&BacktraceFrames[0], BacktraceFrames.size());
  return BacktraceSymbols;
}

} // namespace ur_sanitizer_layer
