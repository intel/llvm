//==---------- host_profiling_info.hpp - SYCL host profiling ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

/// Profiling info for the host execution.
class HostProfilingInfo {
  cl_ulong StartTime = 0;
  cl_ulong EndTime = 0;

public:
  /// Returns event's start time.
  ///
  /// \return event's start time in nanoseconds.
  cl_ulong getStartTime() const { return StartTime; }
  /// Returns event's end time.
  ///
  /// \return event's end time in nanoseconds.
  cl_ulong getEndTime() const { return EndTime; }

  /// Measures event's start time.
  void start();
  /// Measures event's end time.
  void end();
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
