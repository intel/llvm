//===--------- common.hpp - CUDA Adapter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <sycl/detail/defines.hpp>
#include <ur/ur.hpp>

ur_result_t map_error_ur(CUresult result);

/// Converts CUDA error into UR error codes, and outputs error information
/// to stderr.
/// If PI_CUDA_ABORT env variable is defined, it aborts directly instead of
/// throwing the error. This is intended for debugging purposes.
/// \return UR_RESULT_SUCCESS if \param result was CUDA_SUCCESS.
/// \throw ur_result_t exception (integer) if input was not success.
///
ur_result_t check_error_ur(CUresult result, const char *function, int line,
                           const char *file);

#define UR_CHECK_ERROR(result)                                                 \
  check_error_ur(result, __func__, __LINE__, __FILE__)

std::string getCudaVersionString();

constexpr size_t MaxMessageSize = 256;
extern thread_local ur_result_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *message,
                                      ur_result_t error_code);

/// ------ Error handling, matching OpenCL plugin semantics.
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
