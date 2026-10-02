//===--------- ur2offload.hpp - LLVM Offload Adapter ----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <OffloadAPI.h>
#include <unified-runtime/ur_api.h>

inline ur_result_t offloadResultToUR(ol_result_t Result) {
  if (Result == OL_SUCCESS) {
    return UR_RESULT_SUCCESS;
  }

  switch (Result->Code) {
  case OL_ERRC_UNKNOWN:
    return UR_RESULT_ERROR_UNKNOWN;
  case OL_ERRC_HOST_IO:
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  case OL_ERRC_INVALID_BINARY:
    return UR_RESULT_ERROR_INVALID_BINARY;
  case OL_ERRC_INVALID_NULL_HANDLE:
    return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
  case OL_ERRC_INVALID_NULL_POINTER:
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  case OL_ERRC_INVALID_ARGUMENT:
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  case OL_ERRC_NOT_FOUND:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case OL_ERRC_OUT_OF_RESOURCES:
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  case OL_ERRC_INVALID_ENUMERATION:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  case OL_ERRC_INVALID_SIZE:
    return UR_RESULT_ERROR_INVALID_SIZE;
  case OL_ERRC_HOST_TOOL_NOT_FOUND:
    return UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE;
  case OL_ERRC_INVALID_VALUE:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case OL_ERRC_UNIMPLEMENTED:
  case OL_ERRC_UNSUPPORTED:
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  case OL_ERRC_ASSEMBLE_FAILURE:
  case OL_ERRC_COMPILE_FAILURE:
    return UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE;
  case OL_ERRC_LINK_FAILURE:
    return UR_RESULT_ERROR_PROGRAM_LINK_FAILURE;
  case OL_ERRC_BACKEND_FAILURE:
    return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
  case OL_ERRC_UNINITIALIZED:
    return UR_RESULT_ERROR_UNINITIALIZED;
  case OL_ERRC_INVALID_PLATFORM:
    return UR_RESULT_ERROR_INVALID_PLATFORM;
  case OL_ERRC_INVALID_DEVICE:
    return UR_RESULT_ERROR_INVALID_DEVICE;
  case OL_ERRC_INVALID_QUEUE:
    return UR_RESULT_ERROR_INVALID_QUEUE;
  case OL_ERRC_INVALID_EVENT:
    return UR_RESULT_ERROR_INVALID_EVENT;
  case OL_ERRC_INVALID_CONTEXT:
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  case OL_ERRC_SYMBOL_KIND:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case OL_ERRC_SUCCESS:
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}
