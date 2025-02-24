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

void checkErrorUR(ur_result_t Result, const char *Function, int Line,
                  const char *File);

#define UR_CHECK_ERROR(Result)                                                 \
  checkErrorUR(Result, __func__, __LINE__, __FILE__)

std::string getCudaVersionString();

constexpr size_t MaxMessageSize = 256;
extern thread_local ur_result_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage,
                                      ur_result_t ErrorCode);

void setPluginSpecificMessage(CUresult cu_res);

/// ------ Error handling, matching OpenCL plugin semantics.
namespace detail {
namespace ur {

// Report error and no return (keeps compiler from printing warnings).
// TODO: Probably change that to throw a catchable exception,
//       but for now it is useful to see every failure.
//
[[noreturn]] void die(const char *Message);

// Reports error messages
void cuPrint(const char *Message);

void assertion(bool Condition, const char *Message = nullptr);

} // namespace ur
} // namespace detail

namespace umf {

using cuda_params_unique_handle_t = std::unique_ptr<
    umf_cuda_memory_provider_params_t,
    std::function<umf_result_t(umf_cuda_memory_provider_params_handle_t)>>;

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
