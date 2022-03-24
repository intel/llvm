//==------ invoke_simd.hpp - SYCL invoke_simd extension --*- C++ -*---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// Implemenation of the sycl_ext_oneapi_invoke_simd extension.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc
// ===--------------------------------------------------------------------=== //

#pragma once

// 1 - Initial extension version. Base features are supported.
#define SYCL_EXT_ONEAPI_INVOKE_SIMD 1

#include <sycl/ext/oneapi/experimental/uniform.hpp>

#include <std/experimental/simd.hpp>

// TODO FIXME dummy implementation to kick-off testing
constexpr int __builtin_get_reqd_subgroup_size() { return /*FE builtin*/ 16; }

/// Middle End - to - Back End interface to invoke explicit SIMD functions from
/// SPMD SYCL context.
template <class Ret, class F, class... Args>
Ret __builtin_invoke_simd(F, Args...) SYCL_EXTERNAL
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw sycl::exception(sycl::errc::feature_not_supported);
}
#endif // __SYCL_DEVICE_ONLY__

namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

namespace simd_abi {
template <class T, int N>
using native_fixed_size = typename std::experimental::__simd_abi<
    std::experimental::_StorageKind::_VecExt, N>;
} // namespace simd_abi

template <class T, int N>
using simd = std::experimental::simd<T, simd_abi::native_fixed_size<T, N>>;

template <class T, int N>
using simd_mask =
    std::experimental::simd_mask<T, simd_abi::native_fixed_size<T, N>>;

namespace detail {
template <class T, int N = __builtin_get_reqd_subgroup_size(), class = void>
struct spmd2simd;
template <class T, int N> struct spmd2simd<uniform<T>, N> {
  using type = T;
};
template <class... T, int N> struct spmd2simd<std::tuple<T...>, N> {
  using type = std::tuple<typename spmd2simd<T, N>::type...>;
};
template <class T, int N>
struct spmd2simd<T, N, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = simd<T, N>;
};
// TODO implement bool translation

template <class, class = void> struct simd2spmd;
template <class T, int N> struct simd2spmd<simd<T, N>> {
  using type = T;
};
template <class... T> struct simd2spmd<std::tuple<T...>> {
  using type = std::tuple<typename simd2spmd<T>::type...>;
};
template <class T>
struct simd2spmd<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = uniform<T>;
};

// Unwrap the uniform object so that it is passed as underlying type
template <typename T> struct unwrap {
  static auto impl(T val) { return val; }
};

template <typename T> struct unwrap<uniform<T>> {
  static T impl(uniform<T> val) { return val; }
};

template <class Callable, class... T>
using SPMDInvokeResult = typename simd2spmd<
    std::invoke_result_t<Callable, typename spmd2simd<T>::type...>>::type;
} // namespace detail

/// The invoke_simd free function invokes a SIMD function using all work-items
/// in a sub_group. The invoke_simd interface marshals data between the SPMD
/// context of the calling kernel and the SIMD context of the callee, converting
/// arguments and return values between scalar and SIMD types as appropriate.
///
/// @param sg the subgroup simd function is invoked from
/// @param f represents the invoked simd function.
///   Must be a C++ callable that can be invoked with the same number of
///   arguments specified in the args parameter pack. Callable may be a function
///   object, a lambda, or a function pointer (if the device supports
///   SPV_INTEL_function_pointers). Callable must be an immutable callable with
///   the same type and state for all work-items in the sub-group, otherwise
///   behavior is undefined.
/// @param args SPMD parameters to the invoked function, which undergo
///   transformation before actual passing to the simd function, as described in
///   the specification.
// TODO works only for functions now, enable for other callables.
template <class Callable, class... T>
__attribute__((always_inline)) detail::SPMDInvokeResult<Callable, T...>
invoke_simd(sycl::sub_group sg, Callable &&f, T... args) {
  using RetSpmd = detail::SPMDInvokeResult<Callable, T...>;
  return __builtin_invoke_simd<RetSpmd>(f, detail::unwrap<T>::impl(args)...);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
