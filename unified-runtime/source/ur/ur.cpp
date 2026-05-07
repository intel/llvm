
//===--------- ur.cpp - Unified Runtime  ----------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur.hpp"
#include <cassert>

// Controls tracing UR calls from within the UR itself.
bool PrintTrace = [] {
  const char *UrRet = std::getenv("SYCL_UR_TRACE");
  const char *PiRet = std::getenv("SYCL_PI_TRACE");
  const char *Trace = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  int TraceValue = 0;
  if (Trace) {
    try {
      TraceValue = std::stoi(Trace);
    } catch (...) {
      // no-op, we don't have a logger yet to output an error.
    }
  }

  if (TraceValue == -1 || TraceValue == 2) {
    return true;
  }
  return false;
}();
