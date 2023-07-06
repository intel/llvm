//==-------- builtins_utils.hpp - SYCL built-in function utilities ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/generic_type_traits.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

#ifdef __SYCL_DEVICE_ONLY__
#define __sycl_std
#else
namespace __sycl_std = __host_std;
#endif

namespace detail {
#ifdef __FAST_MATH__
template <typename T> struct use_fast_math : is_genfloatf<T> {};
#else
template <typename> struct use_fast_math : std::false_type {};
#endif
template <typename T>
static constexpr bool use_fast_math_v = use_fast_math<T>::value;

// sycl::select(sgentype a, sgentype b, bool c) calls OpenCL built-in
// select(sgentype a, sgentype b, igentype c). This type trait makes the
// proper conversion for argument c from bool to igentype, based on sgentype
// == T.
template <typename T>
using get_select_opencl_builtin_c_arg_type = typename std::conditional_t<
    sizeof(T) == 1, char,
    std::conditional_t<
        sizeof(T) == 2, short,
        std::conditional_t<
            (detail::is_contained<
                 T, detail::type_list<long, unsigned long>>::value &&
             (sizeof(T) == 4 || sizeof(T) == 8)),
            long, // long and ulong are 32-bit on
                  // Windows and 64-bit on Linux
            std::conditional_t<
                sizeof(T) == 4, int,
                std::conditional_t<sizeof(T) == 8, long long, void>>>>>;

template <typename T, typename... Ts> constexpr bool CheckTypeIn() {
  constexpr bool SameType[] = {
      std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<Ts>>...};
  // Replace with std::any_of with C++20.
  for (size_t I = 0; I < sizeof...(Ts); ++I)
    if (SameType[I])
      return true;
  return false;
}

template <typename T, typename... Ts> constexpr bool CheckAllTypesSame() {
  constexpr bool SameType[] = {
      std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<Ts>>...};
  // Replace with std::all_of with C++20.
  for (size_t I = 0; I < sizeof...(Ts); ++I)
    if (!SameType[I])
      return false;
  return true;
}

template <int N, int... Ns> constexpr bool CheckSizeIn() {
  constexpr bool SameSize[] = {(N == Ns)...};
  // Replace with std::any_of with C++20.
  for (size_t I = 0; I < sizeof...(Ns); ++I)
    if (SameSize[I])
      return true;
  return false;
}

template <typename T, typename... Ts>
struct is_valid_vec_type : std::false_type {};
template <typename T, int N, typename... Ts>
struct is_valid_vec_type<vec<T, N>, Ts...>
    : std::bool_constant<CheckTypeIn<T, Ts...>()> {};

template <typename T, int... Ns> struct is_valid_vec_size : std::false_type {};
template <typename T, int N, int... Ns>
struct is_valid_vec_size<vec<T, N>, Ns...>
    : std::bool_constant<CheckSizeIn<N, Ns...>()> {};

template <typename T, typename... Ts>
constexpr bool is_valid_vec_type_v = is_valid_vec_type<T, Ts...>::value;
template <typename T, int... Ns>
constexpr bool is_valid_vec_size_v = is_valid_vec_size<T, Ns...>::value;

template <typename T> struct vec_return;
template <typename T, int N> struct vec_return<vec<T, N>> {
  using type = vec<T, N>;
};
// TODO: Make specialization for swizzle.

template <typename T> using vec_return_t = typename vec_return<T>::type;

template <typename T> struct same_size_signed_int {
  using type = std::conditional_t<
      sizeof(T) == 1, int8_t,
      std::conditional_t<
          sizeof(T) == 2, int16_t,
          std::conditional_t<
              sizeof(T) == 4, int32_t,
              std::conditional_t<sizeof(T) == 8, int64_t, void>>>>;
};

template <typename T, int N> struct same_size_signed_int<vec<T, N>> {
  using type = vec<typename same_size_signed_int<T>::type, N>;
};

template <typename T>
using same_size_signed_int_t = typename same_size_signed_int<T>::type;

template <typename T> struct same_size_unsigned_int {
  using type = std::conditional_t<
      sizeof(T) == 1, uint8_t,
      std::conditional_t<
          sizeof(T) == 2, uint16_t,
          std::conditional_t<
              sizeof(T) == 4, uint32_t,
              std::conditional_t<sizeof(T) == 8, uint64_t, void>>>>;
};

template <typename T, int N> struct same_size_unsigned_int<vec<T, N>> {
  using type = vec<typename same_size_unsigned_int<T>::type, N>;
};

template <typename T>
using same_size_unsigned_int_t = typename same_size_unsigned_int<T>::type;

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
