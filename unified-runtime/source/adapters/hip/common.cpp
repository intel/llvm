//===--------- common.cpp - HIP Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "common.hpp"
#include "logger/ur_logger.hpp"

#include <sstream>

#ifdef SYCL_ENABLE_KERNEL_FUSION
ur_result_t mapErrorUR(amd_comgr_status_t Result) {
  switch (Result) {
  case AMD_COMGR_STATUS_SUCCESS:
    return UR_RESULT_SUCCESS;
  case AMD_COMGR_STATUS_ERROR:
    return UR_RESULT_ERROR_UNKNOWN;
  case AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT:
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  case AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES:
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}
#endif

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
  case hipErrorNotInitialized:
    return UR_RESULT_ERROR_UNINITIALIZED;
  case hipErrorNotSupported:
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

#ifdef SYCL_ENABLE_KERNEL_FUSION
void checkErrorUR(amd_comgr_status_t Result, const char *Function, int Line,
                  const char *File) {
  if (Result == AMD_COMGR_STATUS_SUCCESS) {
    return;
  }

  const char *ErrorString = nullptr;
  const char *ErrorName = nullptr;
  switch (Result) {
  case AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT:
    ErrorName = "AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT";
    ErrorString =
        "One of the actual arguments does not meet a precondition stated in "
        "the documentation of the corresponding formal argument.";
    break;
  case AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES:
    ErrorName = "AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES";
    ErrorString = "Failed to allocate the necessary resources";
    break;
  case AMD_COMGR_STATUS_ERROR:
    [[fallthrough]];
  default:
    ErrorName = "AMD_COMGR_STATUS_ERROR";
    ErrorString = "Generic error";
    break;
  }
  std::stringstream SS;
  SS << "\nUR HIP ERROR:"
     << "\n\tValue:           " << Result
     << "\n\tName:            " << ErrorName
     << "\n\tDescription:     " << ErrorString
     << "\n\tFunction:        " << Function << "\n\tSource Location: " << File
     << ":" << Line << "\n";
  UR_LOG(ERR, "{}", SS.str());

  if (std::getenv("PI_HIP_ABORT") != nullptr ||
      std::getenv("UR_HIP_ABORT") != nullptr) {
    std::abort();
  }

  throw mapErrorUR(Result);
}
#endif

void checkErrorUR(hipError_t Result, const char *Function, int Line,
                  const char *File) {
  if (Result == hipSuccess) {
    return;
  }

  const char *ErrorString = hipGetErrorString(Result);
  const char *ErrorName = hipGetErrorName(Result);

  std::stringstream SS;
  SS << "\nUR HIP ERROR:"
     << "\n\tValue:           " << Result
     << "\n\tName:            " << ErrorName
     << "\n\tDescription:     " << ErrorString
     << "\n\tFunction:        " << Function << "\n\tSource Location: " << File
     << ":" << Line << "\n";
  UR_LOG(ERR, "{}", SS.str());

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

  std::stringstream SS;
  SS << "\nUR HIP ERROR:"
     << "\n\tValue:           " << Result << "\n\tFunction:        " << Function
     << "\n\tSource Location: " << File << ":" << Line << "\n";
  UR_LOG(ERR, "{}", SS.str());

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
