//===--------- common.cpp - HIP Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#include "common.hpp"

#include <sstream>

ur_result_t map_error_ur(hipError_t result) {
  switch (result) {
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

ur_result_t check_error_ur(hipError_t result, const char *function, int line,
                           const char *file) {
  if (result == hipSuccess) {
    return UR_RESULT_SUCCESS;
  }

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") == nullptr) {
    const char *errorString = nullptr;
    const char *errorName = nullptr;
    errorName = hipGetErrorName(result);
    errorString = hipGetErrorString(result);
    std::stringstream ss;
    ss << "\nUR HIP ERROR:"
       << "\n\tValue:           " << result
       << "\n\tName:            " << errorName
       << "\n\tDescription:     " << errorString
       << "\n\tFunction:        " << function << "\n\tSource Location: " << file
       << ":" << line << "\n"
       << std::endl;
    std::cerr << ss.str();
  }

  if (std::getenv("PI_HIP_ABORT") != nullptr) {
    std::abort();
  }

  throw map_error_ur(result);
}

std::string getHipVersionString() {
  int driver_version = 0;
  if (hipDriverGetVersion(&driver_version) != hipSuccess) {
    return "";
  }
  // The version is returned as (1000 major + 10 minor).
  std::stringstream stream;
  stream << "HIP " << driver_version / 1000 << "."
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

void sycl::detail::ur::hipPrint(const char *Message) {
  std::cerr << "ur_print: " << Message << std::endl;
}

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
