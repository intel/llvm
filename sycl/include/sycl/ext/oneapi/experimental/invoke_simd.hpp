//==------ invoke_simd.hpp - SYCL invoke_simd extension --*- C++ -*---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// Implemenation of the SYCL_EXT_ONEAPI_INVOKE_SIMD extension.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/InvokeSIMD/InvokeSIMD.asciidoc
// ===--------------------------------------------------------------------=== //

#pragma once

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
{ throw sycl::exception(sycl::errc::feature_not_supported); }
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
template <class T, int N> struct spmd2simd<uniform<T>, N> { using type = T; };
template <class... T, int N> struct spmd2simd<std::tuple<T...>, N> {
  using type = std::tuple<typename spmd2simd<T, N>::type...>;
};
template <class T, int N>
struct spmd2simd<T, N, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = simd<T, N>;
};
// TODO implement bool translation

template <class, class = void> struct simd2spmd;
template <class T, int N> struct simd2spmd<simd<T, N>> { using type = T; };
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
} // namespace detail

// template<class F, typename O, class...Args>
// typename detail::simd2spmd<std::invoke_result_t<O,
//             typename detail::spmd2simd<Args>::type...>>::type
//__builtin_invoke_simd1(F, O, Args...) SYCL_EXTERNAL;

template <class F, class... Args,
          class R = std::invoke_result_t<
              F, typename detail::spmd2simd<Args>::type...>>
__attribute__((always_inline)) typename detail::simd2spmd<R>::type
invoke_simd(sycl::sub_group sg, F f, Args... args) {
  // TODO works only for function pointers now, enable for other callables.
  using RetSpmd = typename detail::simd2spmd<R>::type;
  return __builtin_invoke_simd<RetSpmd>(f,
                                        detail::unwrap<Args>::impl(args)...);
  // NOTE consider wrapper approach below for other callables
  // (might cause complex data flow for the function pointer):
  // return __builtin_invoke_simd1(
  //    +[](F f1, typename detail::spmd2simd<Args>::type...a){
  //        return f1(a...);
  //    }, f, args...);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
