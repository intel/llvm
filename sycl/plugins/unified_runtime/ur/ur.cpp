
//===--------- ur.cpp - Unified Runtime  ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur.hpp"
#include <cassert>

// Controls tracing UR calls from within the UR itself.
bool PrintTrace = [] {
  const char *PiRet = std::getenv("SYCL_PI_TRACE");
  const char *Trace = PiRet ? PiRet : nullptr;
  const int TraceValue = Trace ? std::stoi(Trace) : 0;
  if (TraceValue == -1 || TraceValue == 2) { // Means print all traces
    return true;
  }
  return false;
}();

// Apparatus for maintaining immutable cache of platforms.
std::vector<ur_platform_handle_t> *URPlatformsCache =
    new std::vector<ur_platform_handle_t>;
SpinLock *URPlatformsCacheMutex = new SpinLock;
bool URPlatformCachePopulated = false;
