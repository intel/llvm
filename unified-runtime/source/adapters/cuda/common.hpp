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
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <umf/providers/provider_cuda.h>

#include "cuda_call_logger.hpp"

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

// Enhanced UR_CHECK_ERROR that automatically logs CUDA calls when tracing is
// enabled
#define UR_CHECK_ERROR(expr)                                                   \
  do {                                                                         \
    if (::ur::cuda::call_logger::isEnabled()) {                                \
      constexpr const char *__call_str = #expr;                                \
      if (__call_str[0] == 'c' && __call_str[1] == 'u') {                      \
        CUDA_CALL_TRACE_GENERIC(__call_str);                                   \
      }                                                                        \
    }                                                                          \
    auto __result = (expr);                                                    \
    if (::ur::cuda::call_logger::isEnabled()) {                                \
      CUDA_CALL_TRACE_RESULT(__result);                                        \
    }                                                                          \
    checkErrorUR(__result, __func__, __LINE__, __FILE__);                      \
  } while (0)

std::string getCudaVersionString();

constexpr size_t MaxMessageSize = 256;
extern thread_local int32_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage, int32_t ErrorCode);

void setPluginSpecificMessage(CUresult cu_res);

namespace umf {
ur_result_t CreateProviderPool(int cuDevice, void *cuContext,
                               umf_usm_memory_type_t type,
                               umf_memory_provider_handle_t *provider,
                               umf_memory_pool_handle_t *pool);
} // namespace umf

namespace ur::cuda {
struct ddi_getter {
  const static ur_dditable_t *value();
};
using handle_base = ur::handle_base<ddi_getter>;
} // namespace ur::cuda
