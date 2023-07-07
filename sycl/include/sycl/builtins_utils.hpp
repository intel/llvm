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
#include <sycl/types.hpp>

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
struct is_valid_elem_type : std::false_type {};
template <typename T, int N, typename... Ts>
struct is_valid_elem_type<vec<T, N>, Ts...>
    : std::bool_constant<CheckTypeIn<T, Ts...>()> {};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes,
          typename... Ts>
struct is_valid_elem_type<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                                    OperationCurrentT, Indexes...>,
                          Ts...>
    : std::bool_constant<CheckTypeIn<typename VecT::element_type, Ts...>()> {};

template <typename T, int... Ns> struct is_valid_size : std::false_type {};
template <typename T, int N, int... Ns>
struct is_valid_size<vec<T, N>, Ns...>
    : std::bool_constant<CheckSizeIn<N, Ns...>()> {};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes,
          int... Ns>
struct is_valid_size<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                               OperationCurrentT, Indexes...>,
                     Ns...>
    : std::bool_constant<CheckSizeIn<sizeof...(Indexes), Ns...>()> {};

template <typename T, typename... Ts>
constexpr bool is_valid_elem_type_v = is_valid_elem_type<T, Ts...>::value;
template <typename T, int... Ns>
constexpr bool is_valid_size_v = is_valid_size<T, Ns...>::value;

template <typename T> struct get_vec;
template <typename T, int N> struct get_vec<vec<T, N>> {
  using type = vec<T, N>;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct get_vec<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                         OperationCurrentT, Indexes...>> {
  using type = vec<typename VecT::element_type, sizeof...(Indexes)>;
};

template <typename T> using get_vec_t = typename get_vec<T>::type;

template <size_t Size> struct get_signed_int_by_size {
  using type = std::conditional_t<
      Size == 1, int8_t,
      std::conditional_t<
          Size == 2, int16_t,
          std::conditional_t<Size == 4, int32_t,
                             std::conditional_t<Size == 8, int64_t, void>>>>;
};

template <size_t Size> struct get_unsigned_int_by_size {
  using type = std::conditional_t<
      Size == 1, uint8_t,
      std::conditional_t<
          Size == 2, uint16_t,
          std::conditional_t<Size == 4, uint32_t,
                             std::conditional_t<Size == 8, uint64_t, void>>>>;
};

template <size_t Size> struct get_float_by_size {
  using type = std::conditional_t<
      Size == 2, half,
      std::conditional_t<Size == 4, float,
                         std::conditional_t<Size == 8, double, void>>>;
};

template <typename T> struct same_size_signed_int {
  using type = typename get_signed_int_by_size<sizeof(T)>::type;
};

template <typename T, int N> struct same_size_signed_int<vec<T, N>> {
  using type = vec<typename same_size_signed_int<T>::type, N>;
};
// TODO: Swizzle variant of this?

template <typename T>
using same_size_signed_int_t = typename same_size_signed_int<T>::type;

template <typename T> struct same_size_unsigned_int {
  using type = typename get_unsigned_int_by_size<sizeof(T)>::type;
};

template <typename T, int N> struct same_size_unsigned_int<vec<T, N>> {
  using type = vec<typename same_size_unsigned_int<T>::type, N>;
};
// TODO: Swizzle variant of this?

template <typename T> struct same_size_float {
  using type = typename get_float_by_size<sizeof(T)>::type;
};

template <typename T, int N> struct same_size_float<vec<T, N>> {
  using type = vec<typename same_size_float<T>::type, N>;
};
// TODO: Swizzle variant of this?

template <typename T>
using same_size_float_t = typename same_size_float<T>::type;

// For upsampling we look for an integer of double the size of the specified
// type.
template <typename T> struct upsampled_int {
  using type =
      std::conditional_t<std::is_unsigned_v<T>,
                         typename get_unsigned_int_by_size<sizeof(T) * 2>::type,
                         typename get_signed_int_by_size<sizeof(T) * 2>::type>;
};
template <typename T, int N> struct upsampled_int<vec<T, N>> {
  using type = vec<typename upsampled_int<T>::type, N>;
};
// TODO: Swizzle variant of this?

template <typename T> using upsampled_int_t = typename upsampled_int<T>::type;

template <typename> struct is_swizzle : std::false_type {};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct is_swizzle<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                            OperationCurrentT, Indexes...>> : std::true_type {};

template <typename T> constexpr bool is_swizzle_v = is_swizzle<T>::value;

template <typename T>
constexpr bool is_vec_or_swizzle_v = is_vec_v<T> || is_swizzle_v<T>;

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
