
//===--------- ur.hpp - Unified Runtime  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "ur.hpp"
#include <cassert>

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
std::vector<ur_platform_handle_t> *PiPlatformsCache =
    new std::vector<ur_platform_handle_t>;
SpinLock *PiPlatformsCacheMutex = new SpinLock;
bool PiPlatformCachePopulated = false;

// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *message,
                                      ur_result_t error_code) {
  assert(strlen(message) <= MaxMessageSize);
  strcpy(ErrorMessage, message);
  ErrorMessageCode = error_code;
}

ur_result_t zerPluginGetLastError(char **message) {
  *message = &ErrorMessage[0];
  return ErrorMessageCode;
}
