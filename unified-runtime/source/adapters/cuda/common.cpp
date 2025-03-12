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
#include "device.hpp"
#include "logger/ur_logger.hpp"
#include "umf_helpers.hpp"

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
  logger::error("{}", SS.str());

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
  logger::error("{}", SS.str());

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

void detail::ur::die(const char *Message) {
  logger::always("ur_die:{}", Message);
  std::terminate();
}

void detail::ur::assertion(bool Condition, const char *Message) {
  if (!Condition)
    die(Message);
}

// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
thread_local ur_result_t ErrorMessageCode = UR_RESULT_SUCCESS;
thread_local char ErrorMessage[MaxMessageSize]{};

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage,
                                      ur_result_t ErrorCode) {
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

ur_result_t getProviderNativeError(const char *, int32_t) {
  // TODO: implement when UMF supports CUDA
  return UR_RESULT_ERROR_UNKNOWN;
}

// Create UMF CUDA memory provider for the host memory (UMF_MEMORY_TYPE_HOST)
// from a device
ur_result_t
createHostMemoryProvider(CUcontext contextCUDA,
                         umf_memory_provider_handle_t *memoryProviderHost) {
  *memoryProviderHost = nullptr;

  umf_cuda_memory_provider_params_handle_t CUMemoryProviderParams = nullptr;
  umf_result_t UmfResult =
      umfCUDAMemoryProviderParamsCreate(&CUMemoryProviderParams);
  UMF_RETURN_UR_ERROR(UmfResult);

  OnScopeExit Cleanup(
      [=]() { umfCUDAMemoryProviderParamsDestroy(CUMemoryProviderParams); });

  UmfResult =
      umf::setCUMemoryProviderParams(CUMemoryProviderParams, 0 /* cuDevice */,
                                     contextCUDA, UMF_MEMORY_TYPE_HOST);
  UMF_RETURN_UR_ERROR(UmfResult);

  // create UMF CUDA memory provider and pool for the host memory
  // (UMF_MEMORY_TYPE_HOST)
  UmfResult = umfMemoryProviderCreate(
      umfCUDAMemoryProviderOps(), CUMemoryProviderParams, memoryProviderHost);
  UMF_RETURN_UR_ERROR(UmfResult);

  return UR_RESULT_SUCCESS;
}

// Create UMF CUDA memory providers for the device memory (UMF_MEMORY_TYPE_HOST)
// and the shared memory (UMF_MEMORY_TYPE_SHARED)
ur_result_t createDeviceMemoryProviders(
    ur_device_handle_t_ *DeviceHandle,
    umf_memory_provider_handle_t *memoryDeviceProvider,
    umf_memory_provider_handle_t *memorySharedProvider) {
  umf_cuda_memory_provider_params_handle_t CUMemoryProviderParams = nullptr;

  umf_result_t UmfResult =
      umfCUDAMemoryProviderParamsCreate(&CUMemoryProviderParams);
  UMF_RETURN_UR_ERROR(UmfResult);

  OnScopeExit Cleanup(
      [=]() { umfCUDAMemoryProviderParamsDestroy(CUMemoryProviderParams); });

  CUdevice device = DeviceHandle->get();
  CUcontext context = DeviceHandle->getNativeContext();

  // create UMF CUDA memory provider for the device memory
  // (UMF_MEMORY_TYPE_DEVICE)
  UmfResult = umf::setCUMemoryProviderParams(CUMemoryProviderParams, device,
                                             context, UMF_MEMORY_TYPE_DEVICE);
  UMF_RETURN_UR_ERROR(UmfResult);

  *memoryDeviceProvider = nullptr;
  UmfResult = umfMemoryProviderCreate(
      umfCUDAMemoryProviderOps(), CUMemoryProviderParams, memoryDeviceProvider);
  UMF_RETURN_UR_ERROR(UmfResult);

  // create UMF CUDA memory provider for the shared memory
  // (UMF_MEMORY_TYPE_SHARED)
  UmfResult = umf::setCUMemoryProviderParams(CUMemoryProviderParams, device,
                                             context, UMF_MEMORY_TYPE_SHARED);
  UMF_RETURN_UR_ERROR(UmfResult);

  *memorySharedProvider = nullptr;
  UmfResult = umfMemoryProviderCreate(
      umfCUDAMemoryProviderOps(), CUMemoryProviderParams, memorySharedProvider);
  UMF_RETURN_UR_ERROR(UmfResult);

  return UR_RESULT_SUCCESS;
}

} // namespace umf
