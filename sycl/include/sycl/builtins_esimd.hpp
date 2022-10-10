//==----------- builtins_esimd.hpp - SYCL ESIMD built-in functions ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/boolean.hpp>
#include <sycl/detail/builtins.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/math_intrin.hpp>
#include <sycl/types.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

// cos
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
cos(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_cos<SZ>(x.data());
#else
  return __esimd_cos<float, SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

// sin
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
sin(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_sin<SZ>(x.data());
#else
  return __esimd_sin<float, SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

// exp
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
exp(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_exp<SZ>(x.data());
#else
  return __esimd_exp<float, SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

// log
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
log(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_log<SZ>(x.data());
#else
  return __esimd_log<float, SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __NOEXC
