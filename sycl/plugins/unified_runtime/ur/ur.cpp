
//===--------- ur.hpp - Unified Runtime  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur.hpp"

// Controls tracing UR calls from within the UR itself.
bool PrintTrace = [] {
  const char *Trace = std::getenv("SYCL_PI_TRACE");
  const int TraceValue = Trace ? std::stoi(Trace) : 0;
  if (TraceValue == -1 || TraceValue == 2) { // Means print all traces
    return true;
  }
  return false;
}();

// Apparatus for maintaining immutable cache of platforms.
std::vector<zer_platform_handle_t> *PiPlatformsCache =
    new std::vector<zer_platform_handle_t>;
SpinLock *PiPlatformsCacheMutex = new SpinLock;
bool PiPlatformCachePopulated = false;
