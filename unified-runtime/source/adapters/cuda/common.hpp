//===--------- common.hpp - CUDA Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <nvml.h>
#include <ur/ur.hpp>

#include <umf/base.h>
#include <umf/providers/provider_cuda.h>

#define UMF_RETURN_UMF_ERROR(UmfResult)                                        \
  do {                                                                         \
    umf_result_t UmfResult_ = (UmfResult);                                     \
    if (UmfResult_ != UMF_RESULT_SUCCESS) {                                    \
      return UmfResult_;                                                       \
    }                                                                          \
  } while (0)

ur_result_t mapErrorUR(CUresult Result);

/// Converts CUDA error into UR error codes, and outputs error information
/// to stderr.
/// If PI_CUDA_ABORT env variable is defined, it aborts directly instead of
/// throwing the error. This is intended for debugging purposes.
/// \return UR_RESULT_SUCCESS if \param Result was CUDA_SUCCESS.
/// \throw ur_result_t exception (integer) if input was not success.
///
void checkErrorUR(CUresult Result, const char *Function, int Line,
                  const char *File);

void checkErrorUR(nvmlReturn_t Result, const char *Function, int Line,
                  const char *File);

void checkErrorUR(ur_result_t Result, const char *Function, int Line,
                  const char *File);

#define UR_CHECK_ERROR(Result)                                                 \
  checkErrorUR(Result, __func__, __LINE__, __FILE__)

std::string getCudaVersionString();

constexpr size_t MaxMessageSize = 256;
extern thread_local int32_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage, int32_t ErrorCode);

void setPluginSpecificMessage(CUresult cu_res);

namespace umf {

inline umf_result_t setCUMemoryProviderParams(
    umf_cuda_memory_provider_params_handle_t CUMemoryProviderParams,
    int cuDevice, void *cuContext, umf_usm_memory_type_t memType) {

  umf_result_t UmfResult =
      umfCUDAMemoryProviderParamsSetContext(CUMemoryProviderParams, cuContext);
  UMF_RETURN_UMF_ERROR(UmfResult);

  UmfResult =
      umfCUDAMemoryProviderParamsSetDevice(CUMemoryProviderParams, cuDevice);
  UMF_RETURN_UMF_ERROR(UmfResult);

  UmfResult =
      umfCUDAMemoryProviderParamsSetMemoryType(CUMemoryProviderParams, memType);
  UMF_RETURN_UMF_ERROR(UmfResult);

  return UMF_RESULT_SUCCESS;
}

} // namespace umf
