//===--------- common.cpp - HIP Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "common.hpp"

#include <sstream>

ur_result_t mapErrorUR(hipError_t Result) {
  switch (Result) {
  case hipSuccess:
    return UR_RESULT_SUCCESS;
  case hipErrorInvalidContext:
    return UR_RESULT_ERROR_INVALID_CONTEXT;
  case hipErrorInvalidDevice:
    return UR_RESULT_ERROR_INVALID_DEVICE;
  case hipErrorInvalidValue:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case hipErrorOutOfMemory:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case hipErrorLaunchOutOfResources:
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

void checkErrorUR(hipError_t Result, const char *Function, int Line,
                  const char *File) {
  if (Result == hipSuccess) {
    return;
  }

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") == nullptr ||
      std::getenv("UR_SUPPRESS_ERROR_MESSAGE") == nullptr) {
    const char *ErrorString = nullptr;
    const char *ErrorName = nullptr;
    ErrorName = hipGetErrorName(Result);
    ErrorString = hipGetErrorString(Result);
    std::cerr << "\nUR HIP ERROR:"
              << "\n\tValue:           " << Result
              << "\n\tName:            " << ErrorName
              << "\n\tDescription:     " << ErrorString
              << "\n\tFunction:        " << Function
              << "\n\tSource Location: " << File << ":" << Line << "\n\n";
  }

  if (std::getenv("PI_HIP_ABORT") != nullptr ||
      std::getenv("UR_HIP_ABORT") != nullptr) {
    std::abort();
  }

  throw mapErrorUR(Result);
}

void checkErrorUR(ur_result_t Result, const char *Function, int Line,
                  const char *File) {
  if (Result == UR_RESULT_SUCCESS) {
    return;
  }

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") == nullptr ||
      std::getenv("UR_SUPPRESS_ERROR_MESSAGE") == nullptr) {
    std::cerr << "\nUR HIP ERROR:"
              << "\n\tValue:           " << Result
              << "\n\tFunction:        " << Function
              << "\n\tSource Location: " << File << ":" << Line << "\n\n";
  }

  if (std::getenv("PI_HIP_ABORT") != nullptr ||
      std::getenv("UR_HIP_ABORT") != nullptr) {
    std::abort();
  }

  throw Result;
}

hipError_t getHipVersionString(std::string &Version) {
  int DriverVersion = 0;
  auto Result = hipDriverGetVersion(&DriverVersion);
  if (Result != hipSuccess) {
    return Result;
  }
  // The version is returned as (1000 major + 10 minor).
  std::stringstream Stream;
  Stream << "HIP " << DriverVersion / 1000 << "." << DriverVersion % 1000 / 10;
  Version = Stream.str();
  return Result;
}

void detail::ur::die(const char *pMessage) {
  std::cerr << "ur_die: " << pMessage << '\n';
  std::terminate();
}

void detail::ur::assertion(bool Condition, const char *pMessage) {
  if (!Condition)
    die(pMessage);
}

void detail::ur::hipPrint(const char *pMessage) {
  std::cerr << "ur_print: " << pMessage << '\n';
}

// Global variables for UR_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage,
                                      ur_result_t ErrorCode) {
  assert(strlen(pMessage) < MaxMessageSize);
  strncpy(ErrorMessage, pMessage, MaxMessageSize - 1);
  ErrorMessageCode = ErrorCode;
}

// Returns plugin specific error and warning messages; common implementation
// that can be shared between adapters
ur_result_t urGetLastResult(ur_platform_handle_t, const char **ppMessage) {
  *ppMessage = &ErrorMessage[0];
  return ErrorMessageCode;
}
