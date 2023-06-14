//===--------- common.cpp - CUDA Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

#include <cuda.h>

#include <sstream>

ur_result_t mapErrorUR(CUresult Result) {
  switch (Result) {
  case CUDA_SUCCESS:
    return UR_RESULT_SUCCESS;
  case CUDA_ERROR_NOT_PERMITTED:
    return UR_RESULT_ERROR_INVALID_OPERATION;
  case CUDA_ERROR_INVALID_CONTEXT:
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  case CUDA_ERROR_INVALID_DEVICE:
    return UR_RESULT_ERROR_INVALID_DEVICE;
  case CUDA_ERROR_INVALID_VALUE:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case CUDA_ERROR_OUT_OF_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

ur_result_t checkErrorUR(CUresult Result, const char *Function, int Line,
                         const char *File) {
  if (Result == CUDA_SUCCESS || Result == CUDA_ERROR_DEINITIALIZED) {
    return UR_RESULT_SUCCESS;
  }

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") == nullptr) {
    const char *ErrorString = nullptr;
    const char *ErrorName = nullptr;
    cuGetErrorName(Result, &ErrorName);
    cuGetErrorString(Result, &ErrorString);
    std::stringstream SS;
    SS << "\nUR CUDA ERROR:"
       << "\n\tValue:           " << Result
       << "\n\tName:            " << ErrorName
       << "\n\tDescription:     " << ErrorString
       << "\n\tFunction:        " << Function << "\n\tSource Location: " << File
       << ":" << Line << "\n"
       << std::endl;
    std::cerr << SS.str();
  }

  if (std::getenv("PI_CUDA_ABORT") != nullptr) {
    std::abort();
  }

  throw mapErrorUR(Result);
}

std::string getCudaVersionString() {
  int driver_version = 0;
  cuDriverGetVersion(&driver_version);
  // The version is returned as (1000 major + 10 minor).
  std::stringstream stream;
  stream << "CUDA " << driver_version / 1000 << "."
         << driver_version % 1000 / 10;
  return stream.str();
}

void sycl::detail::ur::die(const char *Message) {
  std::cerr << "ur_die: " << Message << std::endl;
  std::terminate();
}

void sycl::detail::ur::assertion(bool Condition, const char *Message) {
  if (!Condition)
    die(Message);
}

void sycl::detail::ur::cuPrint(const char *Message) {
  std::cerr << "ur_print: " << Message << std::endl;
}

// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage,
                                      ur_result_t ErrorCode) {
  assert(strlen(pMessage) <= MaxMessageSize);
  strcpy(ErrorMessage, pMessage);
  ErrorMessageCode = ErrorCode;
}

// Returns plugin specific error and warning messages; common implementation
// that can be shared between adapters
ur_result_t urGetLastResult(ur_platform_handle_t, const char **ppMessage) {
  *ppMessage = &ErrorMessage[0];
  return ErrorMessageCode;
}
