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

ur_result_t map_error_ur(CUresult result) {
  switch (result) {
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

ur_result_t check_error_ur(CUresult result, const char *function, int line,
                           const char *file) {
  if (result == CUDA_SUCCESS || result == CUDA_ERROR_DEINITIALIZED) {
    return UR_RESULT_SUCCESS;
  }

  if (std::getenv("SYCL_PI_SUPPRESS_ERROR_MESSAGE") == nullptr) {
    const char *errorString = nullptr;
    const char *errorName = nullptr;
    cuGetErrorName(result, &errorName);
    cuGetErrorString(result, &errorString);
    std::stringstream ss;
    ss << "\nUR CUDA ERROR:"
       << "\n\tValue:           " << result
       << "\n\tName:            " << errorName
       << "\n\tDescription:     " << errorString
       << "\n\tFunction:        " << function << "\n\tSource Location: " << file
       << ":" << line << "\n"
       << std::endl;
    std::cerr << ss.str();
  }

  if (std::getenv("PI_CUDA_ABORT") != nullptr) {
    std::abort();
  }

  throw map_error_ur(result);
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
