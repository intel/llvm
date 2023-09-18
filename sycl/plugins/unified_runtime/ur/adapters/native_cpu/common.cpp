//===---------------- common.cpp - Native CPU Adapter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

// Global variables for UR_RESULT_ADAPTER_SPECIFIC_ERROR
// See urGetLastResult
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage,
                                      ur_result_t ErrorCode) {
  assert(strlen(pMessage) <= MaxMessageSize);
  strcpy(ErrorMessage, pMessage);
  ErrorMessageCode = ErrorCode;
}

ur_result_t urGetLastResult(ur_platform_handle_t, const char **ppMessage) {
  *ppMessage = &ErrorMessage[0];
  return ErrorMessageCode;
}

void detail::ur::die(const char *pMessage) {
  std::cerr << "ur_die: " << pMessage << '\n';
  std::terminate();
}
