//==--- builtins_utils_vec.hpp - SYCL built-in function utilities for vec --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/builtins_utils_scalar.hpp>

#include <sycl/marray.hpp> // for marray
#include <sycl/types.hpp>  // for vec

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename> struct is_swizzle : std::false_type {};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct is_swizzle<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                            OperationCurrentT, Indexes...>> : std::true_type {};

template <typename T> constexpr bool is_swizzle_v = is_swizzle<T>::value;

template <typename T>
constexpr bool is_vec_or_swizzle_v = is_vec_v<T> || is_swizzle_v<T>;

// Utility trait for checking if T's element type is in Ts.
template <typename T, size_t N, typename... Ts>
struct is_valid_elem_type<marray<T, N>, Ts...>
    : std::bool_constant<check_type_in_v<T, Ts...>> {};
template <typename T, int N, typename... Ts>
struct is_valid_elem_type<vec<T, N>, Ts...>
    : std::bool_constant<check_type_in_v<T, Ts...>> {};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes,
          typename... Ts>
struct is_valid_elem_type<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                    OperationCurrentT, Indexes...>,
                          Ts...>
    : std::bool_constant<check_type_in_v<typename VecT::element_type, Ts...>> {
};
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress, typename... Ts>
struct is_valid_elem_type<multi_ptr<ElementType, Space, DecorateAddress>, Ts...>
    : std::bool_constant<check_type_in_v<ElementType, Ts...>> {};

// Utility trait for getting the number of elements in T.
template <typename T>
struct num_elements : std::integral_constant<size_t, 1> {};
template <typename T, size_t N>
struct num_elements<marray<T, N>> : std::integral_constant<size_t, N> {};
template <typename T, int N>
struct num_elements<vec<T, N>> : std::integral_constant<size_t, size_t(N)> {};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct num_elements<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                              OperationCurrentT, Indexes...>>
    : std::integral_constant<size_t, sizeof...(Indexes)> {};

// Utilty trait for checking that the number of elements in T is in Ns.
template <typename T, size_t... Ns>
struct is_valid_size
    : std::bool_constant<check_size_in_v<num_elements<T>::value, Ns...>> {};

template <typename T, int... Ns>
constexpr bool is_valid_size_v = is_valid_size<T, Ns...>::value;

// Utility for converting a swizzle to a vector or preserve the type if it isn't
// a swizzle.
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct simplify_if_swizzle<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                     OperationCurrentT, Indexes...>> {
  using type = vec<typename VecT::element_type, sizeof...(Indexes)>;
};

template <typename T1, typename T2>
struct is_same_op<
    T1, T2,
    std::enable_if_t<is_vec_or_swizzle_v<T1> && is_vec_or_swizzle_v<T2>>>
    : std::is_same<simplify_if_swizzle_t<T1>, simplify_if_swizzle_t<T2>> {};

template <typename T, size_t N> struct same_size_signed_int<marray<T, N>> {
  using type = marray<typename same_size_signed_int<T>::type, N>;
};
template <typename T, int N> struct same_size_signed_int<vec<T, N>> {
  using type = vec<typename same_size_signed_int<T>::type, N>;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct same_size_signed_int<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                      OperationCurrentT, Indexes...>> {
  // Converts to vec for simplicity.
  using type =
      vec<typename same_size_signed_int<typename VecT::element_type>::type,
          sizeof...(Indexes)>;
};

template <typename T, size_t N> struct same_size_unsigned_int<marray<T, N>> {
  using type = marray<typename same_size_unsigned_int<T>::type, N>;
};
template <typename T, int N> struct same_size_unsigned_int<vec<T, N>> {
  using type = vec<typename same_size_unsigned_int<T>::type, N>;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct same_size_unsigned_int<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                        OperationCurrentT, Indexes...>> {
  // Converts to vec for simplicity.
  using type =
      vec<typename same_size_unsigned_int<typename VecT::element_type>::type,
          sizeof...(Indexes)>;
};

// Utility trait for changing the element type of a type T. If T is a scalar,
// the new type replaces T completely.
template <typename NewElemT, typename T> struct change_elements {
  using type = NewElemT;
};
template <typename NewElemT, typename T, size_t N>
struct change_elements<NewElemT, marray<T, N>> {
  using type = marray<typename change_elements<NewElemT, T>::type, N>;
};
template <typename NewElemT, typename T, int N>
struct change_elements<NewElemT, vec<T, N>> {
  using type = vec<typename change_elements<NewElemT, T>::type, N>;
};
template <typename NewElemT, typename VecT, typename OperationLeftT,
          typename OperationRightT, template <typename> class OperationCurrentT,
          int... Indexes>
struct change_elements<NewElemT,
                       SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                 OperationCurrentT, Indexes...>> {
  // Converts to vec for simplicity.
  using type =
      vec<typename change_elements<NewElemT, typename VecT::element_type>::type,
          sizeof...(Indexes)>;
};

template <typename NewElemT, typename T>
using change_elements_t = typename change_elements<NewElemT, T>::type;

template <typename T> using int_elements_t = change_elements_t<int, T>;
template <typename T> using bool_elements_t = change_elements_t<bool, T>;

template <typename T, size_t N> struct upsampled_int<marray<T, N>> {
  using type = marray<typename upsampled_int<T>::type, N>;
};
template <typename T, int N> struct upsampled_int<vec<T, N>> {
  using type = vec<typename upsampled_int<T>::type, N>;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct upsampled_int<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                               OperationCurrentT, Indexes...>> {
  // Converts to vec for simplicity.
  using type = vec<typename upsampled_int<typename VecT::element_type>::type,
                   sizeof...(Indexes)>;
};

// Wrapper trait around nan_return to allow propagation through swizzles and
// marrays.
template <typename T, size_t N> struct nan_return_unswizzled<marray<T, N>> {
  using type = marray<typename nan_return_unswizzled<T>::type, N>;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct nan_return_unswizzled<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                       OperationCurrentT, Indexes...>> {
  using type = typename nan_return_unswizzled<
      vec<typename VecT::element_type, sizeof...(Indexes)>>::type;
};

// Utility functions for converting to/from vec/marray.
template <class T, size_t N> vec<T, 2> to_vec2(marray<T, N> X, size_t Start) {
  return {X[Start], X[Start + 1]};
}
template <class T, size_t N> vec<T, N> to_vec(marray<T, N> X) {
  vec<T, N> Vec;
  for (size_t I = 0; I < N; I++)
    Vec[I] = X[I];
  return Vec;
}
template <class T, int N> marray<T, N> to_marray(vec<T, N> X) {
  marray<T, N> Marray;
  for (size_t I = 0; I < N; I++)
    Marray[I] = X[I];
  return Marray;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
