//===--------- common.cpp - CUDA Adapter ----------------------------------===//
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

#include "umf_helpers.hpp"

#include <cuda.h>
#include <nvml.h>

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

ur_result_t mapErrorUR(nvmlReturn_t Result) {
  switch (Result) {
  case NVML_SUCCESS:
    return UR_RESULT_SUCCESS;
  case NVML_ERROR_NOT_SUPPORTED:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case NVML_ERROR_GPU_IS_LOST:
    return UR_RESULT_ERROR_DEVICE_LOST;
  case NVML_ERROR_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case NVML_ERROR_INSUFFICIENT_RESOURCES:
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  }
}

void checkErrorUR(CUresult Result, const char *Function, int Line,
                  const char *File) {
  if (Result == CUDA_SUCCESS || Result == CUDA_ERROR_DEINITIALIZED) {
    return;
  }

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
     << ":" << Line << "\n";
  UR_LOG(ERR, "{}", SS.str());

  if (std::getenv("PI_CUDA_ABORT") != nullptr ||
      std::getenv("UR_CUDA_ABORT") != nullptr) {
    std::abort();
  }

  throw mapErrorUR(Result);
}

void checkErrorUR(nvmlReturn_t Result, const char *Function, int Line,
                  const char *File) {
  if (Result == NVML_SUCCESS) {
    return;
  }

  const char *ErrorString = nullptr;
  ErrorString = nvmlErrorString(Result);
  std::stringstream SS;
  SS << "\nUR NVML ERROR:"
     << "\n\tValue:           " << Result
     << "\n\tDescription:     " << ErrorString
     << "\n\tFunction:        " << Function << "\n\tSource Location: " << File
     << ":" << Line << "\n";
  UR_LOG(ERR, "{}", SS.str());

  if (std::getenv("PI_CUDA_ABORT") != nullptr ||
      std::getenv("UR_CUDA_ABORT") != nullptr) {
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
  SS << "\nUR ERROR:"
     << "\n\tValue:           " << Result << "\n\tFunction:        " << Function
     << "\n\tSource Location: " << File << ":" << Line << "\n";
  UR_LOG(ERR, "{}", SS.str());

  if (std::getenv("PI_CUDA_ABORT") != nullptr) {
    std::abort();
  }

  throw Result;
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

// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
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

void setPluginSpecificMessage(CUresult cu_res) {
  const char *error_string;
  const char *error_name;
  cuGetErrorName(cu_res, &error_name);
  cuGetErrorString(cu_res, &error_string);
  char *message = (char *)malloc(strlen(error_string) + strlen(error_name) + 2);
  strcpy(message, error_name);
  strcat(message, "\n");
  strcat(message, error_string);

  setErrorMessage(message, UR_RESULT_ERROR_ADAPTER_SPECIFIC);
  free(message);
}

namespace umf {
ur_result_t getProviderNativeError(const char *providerName, int32_t error) {
  if (strcmp(providerName, "CUDA") == 0) {
    return mapErrorUR(static_cast<CUresult>(error));
  }

  return UR_RESULT_ERROR_UNKNOWN;
}

ur_result_t CreateProviderPool(int cuDevice, void *cuContext,
                               umf_usm_memory_type_t type,
                               umf_memory_provider_handle_t *provider,
                               umf_memory_pool_handle_t *pool) {
  umf_cuda_memory_provider_params_handle_t CUMemoryProviderParams = nullptr;
  UMF_RETURN_UR_ERROR(
      umfCUDAMemoryProviderParamsCreate(&CUMemoryProviderParams));
  OnScopeExit Cleanup(
      [=]() { umfCUDAMemoryProviderParamsDestroy(CUMemoryProviderParams); });

  // Setup memory provider parameters
  UMF_RETURN_UR_ERROR(
      umfCUDAMemoryProviderParamsSetContext(CUMemoryProviderParams, cuContext));
  UMF_RETURN_UR_ERROR(
      umfCUDAMemoryProviderParamsSetDevice(CUMemoryProviderParams, cuDevice));
  UMF_RETURN_UR_ERROR(
      umfCUDAMemoryProviderParamsSetMemoryType(CUMemoryProviderParams, type));

  // Create memory provider
  UMF_RETURN_UR_ERROR(umfMemoryProviderCreate(
      umfCUDAMemoryProviderOps(), CUMemoryProviderParams, provider));

  // Create memory pool
  UMF_RETURN_UR_ERROR(
      umfPoolCreate(umfProxyPoolOps(), *provider, nullptr, 0, pool));

  return UR_RESULT_SUCCESS;
}
} // namespace umf
