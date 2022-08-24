//===-------- error_handling.hpp - SYCL error handling  ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_impl.hpp>
#include <sycl/detail/cg.hpp>
#include <sycl/detail/pi.h>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

namespace enqueue_kernel_launch {
/// Analyzes error code and arguments of piEnqueueKernelLaunch to emit
/// user-friendly exception describing the problem.
///
/// This function is expected to be called only for non-success error codes,
/// i.e. the first argument must not be equal to PI_SUCCESS.
///
/// This function actually never returns and always throws an exception with
/// error description.
void handleErrorOrWarning(pi_result, const device_impl &, pi_kernel,
                          const NDRDescT &);
} // namespace enqueue_kernel_launch

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
