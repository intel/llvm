//==-------- clock.hpp --- SYCL extension for clock() free function --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/aspects.hpp>
#include <sycl/exception.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class clock_scope : int {
  // Aligned with SPIR-V Scope<id> values.
  device = 1,
  work_group = 2,
  sub_group = 3
};

namespace detail {
template <clock_scope Scope> inline uint64_t clock_impl() {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
  // Currently clock() is not supported on NVPTX and AMDGCN.
  return 0;
#else
  return __spirv_ReadClockKHR(static_cast<int>(Scope));
#endif // defined(__NVPTX__) || defined(__AMDGCN__)
#else
  throw sycl::exception(
      make_error_code(errc::runtime),
      "sycl::ext::oneapi::experimental::clock() is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
}
} // namespace detail

template <clock_scope Scope> inline uint64_t clock();

// Specialization for device.
template <>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_clock_device)]]
#endif
inline uint64_t clock<clock_scope::device>() {
  return detail::clock_impl<clock_scope::device>();
}

// Specialization for work-group.
template <>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_clock_work_group)]]
#endif
inline uint64_t clock<clock_scope::work_group>() {
  return detail::clock_impl<clock_scope::work_group>();
}

// Specialization for sub-group.
template <>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::__uses_aspects__(sycl::aspect::ext_oneapi_clock_sub_group)]]
#endif
inline uint64_t clock<clock_scope::sub_group>() {
  return detail::clock_impl<clock_scope::sub_group>();
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
