//==---------- Forward declarations and traits for vector/marray types -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <type_traits>

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
template <typename DataT, int NumElements> class __SYCL_EBO vec;

template <typename DataT, std::size_t N> class marray;

namespace detail {
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
class SwizzleOp;

// --------- is_* traits ------------------ //
template <typename> struct is_vec : std::false_type {};
template <typename T, int N> struct is_vec<vec<T, N>> : std::true_type {};
template <typename T> constexpr bool is_vec_v = is_vec<T>::value;

template <typename T, typename = void>
struct is_ext_vector : std::false_type {};
#if defined(__has_extension)
#if __has_extension(attribute_ext_vector_type)
template <typename T, int N>
struct is_ext_vector<T __attribute__((ext_vector_type(N)))> : std::true_type {};
#endif
#endif
template <typename T>
inline constexpr bool is_ext_vector_v = is_ext_vector<T>::value;

template <typename> struct is_swizzle : std::false_type {};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct is_swizzle<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                            OperationCurrentT, Indexes...>> : std::true_type {};
template <typename T> constexpr bool is_swizzle_v = is_swizzle<T>::value;

template <typename T>
constexpr bool is_vec_or_swizzle_v = is_vec_v<T> || is_swizzle_v<T>;

template <typename> struct is_marray : std::false_type {};
template <typename T, std::size_t N>
struct is_marray<marray<T, N>> : std::true_type {};
template <typename T> constexpr bool is_marray_v = is_marray<T>::value;

// --------- num_elements trait ------------------ //
template <typename T>
struct num_elements : std::integral_constant<std::size_t, 1> {};
template <typename T, std::size_t N>
struct num_elements<marray<T, N>> : std::integral_constant<std::size_t, N> {};
template <typename T, int N>
struct num_elements<vec<T, N>>
    : std::integral_constant<std::size_t, std::size_t(N)> {};
#if defined(__has_extension)
#if __has_extension(attribute_ext_vector_type)
template <typename T, int N>
struct num_elements<T __attribute__((ext_vector_type(N)))>
    : std::integral_constant<std::size_t, N> {};
#endif
#endif
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct num_elements<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                              OperationCurrentT, Indexes...>>
    : std::integral_constant<std::size_t, sizeof...(Indexes)> {};

template <typename T>
inline constexpr std::size_t num_elements_v = num_elements<T>::value;

// --------- element_type trait ------------------ //
template <typename T, typename = void> struct element_type {
  using type = T;
};
template <typename T, int N> struct element_type<vec<T, N>> {
  using type = T;
};
template <typename T, std::size_t N> struct element_type<marray<T, N>> {
  using type = T;
};
#if defined(__has_extension)
#if __has_extension(attribute_ext_vector_type)
template <typename T, int N>
struct element_type<T __attribute__((ext_vector_type(N)))> {
  using type = T;
};
#endif
#endif
template <typename T> using element_type_t = typename element_type<T>::type;

template <int N>
inline constexpr bool is_allowed_vec_size_v =
    N == 1 || N == 2 || N == 3 || N == 4 || N == 8 || N == 16;

} // namespace detail
} // namespace _V1
} // namespace sycl
