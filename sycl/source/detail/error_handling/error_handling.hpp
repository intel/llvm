//===-------- error_handling.hpp - SYCL error handling  ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/cg.hpp>
#include <detail/device_impl.hpp>
#include <ur_api.h>

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace enqueue_kernel_launch {
/// Analyzes error code and arguments of urEnqueueKernelLaunch to emit
/// user-friendly exception describing the problem.
///
/// This function is expected to be called only for non-success error codes,
/// i.e. the first argument must not be equal to UR_RESULT_SUCCESS.
///
/// This function actually never returns and always throws an exception with
/// error description.
void handleErrorOrWarning(ur_result_t, const device_impl &, ur_kernel_handle_t,
                          const NDRDescT &);
} // namespace enqueue_kernel_launch

namespace kernel_get_group_info {
/// Analyzes error code of urKernelGetGroupInfo.
void handleErrorOrWarning(ur_result_t, ur_kernel_group_info_t,
                          const AdapterPtr &);
} // namespace kernel_get_group_info

} // namespace detail
} // namespace _V1
} // namespace sycl
