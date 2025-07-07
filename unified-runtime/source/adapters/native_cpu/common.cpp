//===---------------- common.cpp - Native CPU Adapter ---------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

// Global variables for UR_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local int32_t ErrorMessageCode = 0;
thread_local char ErrorMessage[MaxMessageSize]{};

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage, int32_t ErrorCode) {
  assert(strlen(pMessage) < MaxMessageSize);
  // Copy at most MaxMessageSize - 1 bytes to ensure the resultant string is
  // always null terminated.
  strncpy(ErrorMessage, pMessage, MaxMessageSize - 1);
  ErrorMessageCode = ErrorCode;
}
