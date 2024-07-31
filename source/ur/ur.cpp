
//===--------- ur.cpp - Unified Runtime  ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
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
  const int TraceValue = Trace ? std::stoi(Trace) : 0;
  if ((PiRet && (TraceValue == -1 || TraceValue == 2)) ||
      (UrRet && TraceValue != 0)) {
    return true;
  }
  return false;
}();
