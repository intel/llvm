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

// Unwrap the uniform object so that it is passed as unerlying type
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
//__builtin_invoke_simd(F, O, Args...) SYCL_EXTERNAL;

template <class Ret, class F, class... Args>
Ret __builtin_invoke_simd1(F, Args...) SYCL_EXTERNAL;

template <class F, class... Args,
          class R = std::invoke_result_t<
              F, typename detail::spmd2simd<Args>::type...>>
__attribute__((always_inline)) typename detail::simd2spmd<R>::type
invoke_simd(sycl::sub_group sg, F f, Args... args) {
  constexpr bool is_func_ptr = std::is_function_v<std::remove_pointer_t<F>>;
  if constexpr (is_func_ptr) {
    // this branch works only for function pointers
    using RetSpmd = typename detail::simd2spmd<R>::type;
    return __builtin_invoke_simd1<RetSpmd>(f,
                                           detail::unwrap<Args>::impl(args)...);
  } else {
    static_assert(
        is_func_ptr,
        "Only function pointers are supported by invoke_simd for now");
    // NOTE using the same approach with wrapper lambda leads to complex
    // function pointer data flow, which may cause
    // return __builtin_invoke_simd(
    //    +[](F f1, typename detail::spmd2simd<Args>::type...a){
    //        return f1(a...);
    //    }, f, args...);
    // TODO enable this for other callables.
  }
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
