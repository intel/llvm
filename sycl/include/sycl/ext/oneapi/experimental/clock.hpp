//==-------- clock.hpp --- SYCL extension for clock() free function --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/exception.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class clock_scope : int {
  // Aligned with SPIR-V Scope<id> values
  device = 1,
  work_group = 2,
  sub_group = 3
};

#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_clock)]]
#endif // __SYCL_DEVICE_ONLY__
uint64_t clock(clock_scope scope = clock_scope::sub_group) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_ReadClockKHR(static_cast<int>(scope));
#else
  throw sycl::exception(
      make_error_code(errc::runtime),
      "sycl::ext::oneapi::experimental::clock() is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
