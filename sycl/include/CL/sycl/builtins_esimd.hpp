//==----------- builtins_esimd.hpp - SYCL ESIMD built-in functions ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/detail/boolean.hpp>
#include <sycl/__impl/detail/builtins.hpp>
#include <sycl/__impl/detail/common.hpp>
#include <sycl/__impl/detail/generic_type_traits.hpp>
#include <sycl/__impl/types.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/math_intrin.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

#ifdef __SYCL_ENABLE_SYCL121_NAMESPACE
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
#else
namespace __sycl_internal {
inline namespace __v1 {
#endif

// cos
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
cos(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_cos<SZ>(x.data());
#else
  return __esimd_cos<SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

// sin
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
sin(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_sin<SZ>(x.data());
#else
  return __esimd_sin<SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

// exp
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
exp(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_exp<SZ>(x.data());
#else
  return __esimd_exp<SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

// log
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE __ESIMD_NS::simd<float, SZ>
log(__ESIMD_NS::simd<float, SZ> x) __NOEXC {
#ifdef __SYCL_DEVICE_ONLY__
  return __ESIMD_NS::detail::ocl_log<SZ>(x.data());
#else
  return __esimd_log<SZ>(x.data());
#endif // __SYCL_DEVICE_ONLY__
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#undef __NOEXC

#ifndef __SYCL_ENABLE_SYCL121_NAMESPACE
namespace sycl {
  using namespace __sycl_internal::__v1;
}
#endif
