//===--------- ur2offload.hpp - LLVM Offload Adapter ----------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <ur_api.h>

inline ur_result_t offloadResultToUR(ol_result_t Result) {
  if (Result == OL_SUCCESS) {
    return UR_RESULT_SUCCESS;
  }

  switch (Result->Code) {
  case OL_ERRC_INVALID_NULL_HANDLE:
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  case OL_ERRC_INVALID_NULL_POINTER:
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  case OL_ERRC_UNSUPPORTED:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}
