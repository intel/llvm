//==---------- host_profiling_info.hpp - SYCL host profiling ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp> // for __SYCL_EXPORT

#include <stdint.h> // for uint64_t

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Profiling info for the host execution.
class __SYCL_EXPORT HostProfilingInfo {
  uint64_t StartTime = 0;
  uint64_t EndTime = 0;

public:
  /// Returns event's start time.
  ///
  /// \return event's start time in nanoseconds.
  uint64_t getStartTime() const { return StartTime; }
  /// Returns event's end time.
  ///
  /// \return event's end time in nanoseconds.
  uint64_t getEndTime() const { return EndTime; }

  /// Measures event's start time.
  void start();
  /// Measures event's end time.
  void end();
};
} // namespace detail
} // namespace _V1
} // namespace sycl
