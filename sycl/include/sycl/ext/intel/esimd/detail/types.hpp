//==-------------- types.hpp - DPC++ Explicit SIMD API ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Types and type traits to implement Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/detail/stl_type_traits.hpp> // to define C++14,17 extensions
#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/region.hpp>
#include <sycl/ext/intel/esimd/detail/types_elementary.hpp>
#include <sycl/half_type.hpp>

#if defined(__ESIMD_DBG_HOST) && !defined(__SYCL_DEVICE_ONLY__)
#define __esimd_dbg_print(a) std::puts(">>> " #a);
#else
#define __esimd_dbg_print(a)
#endif // defined(__ESIMD_DBG_HOST) && !defined(__SYCL_DEVICE_ONLY__)

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd {

// simd and simd_view_impl forward declarations
template <typename Ty, int N> class simd;
template <typename BaseTy, typename RegionTy> class simd_view;

/// @cond ESIMD_DETAIL

namespace detail {

// forward declarations of major internal simd classes
template <typename Ty, int N> class simd_mask_impl;
template <typename RawT, int N, class Derived, class SFINAE = void>
class simd_obj_impl;

// @{
// Helpers for major simd classes, which don't require their definitions to
// compile. Error checking/SFINAE is not used as these are only used internally.

using simd_mask_elem_type = unsigned short;
template <int N> using simd_mask_type = simd_mask_impl<simd_mask_elem_type, N>;

template <typename T>
static inline constexpr bool is_esimd_scalar_v = std::is_arithmetic_v<T>;

template <typename T>
using is_esimd_scalar = typename std::bool_constant<is_esimd_scalar_v<T>>;

// @{
// Checks if given type T derives from simd_obj_impl or is equal to it.
template <typename T>
struct is_simd_obj_impl_derivative : public std::false_type {};

// Specialization for the simd_obj_impl type itself.
template <typename RawT, int N, class Derived>
struct is_simd_obj_impl_derivative<simd_obj_impl<RawT, N, Derived>>
    : public std::true_type {};

// Specialization for all other types.
template <typename T, int N, template <typename, int> class Derived>
struct is_simd_obj_impl_derivative<Derived<T, N>>
    : public std::conditional_t<
          std::is_base_of_v<simd_obj_impl<__raw_t<T>, N, Derived<T, N>>,
                            Derived<T, N>>,
          std::true_type, std::false_type> {};

// Convenience shortcut.
template <typename T>
inline constexpr bool is_simd_obj_impl_derivative_v =
    is_simd_obj_impl_derivative<T>::value;
// @}

// @{
// "Resizes" given simd type \c T to given number of elements \c N.
template <class SimdT, int Ndst> struct resize_a_simd_type;

// Specialization for the simd_obj_impl type.
template <typename T, int Nsrc, int Ndst, template <typename, int> class SimdT>
struct resize_a_simd_type<simd_obj_impl<__raw_t<T>, Nsrc, SimdT<T, Nsrc>>,
                          Ndst> {
  using type = simd_obj_impl<__raw_t<T>, Ndst, SimdT<T, Ndst>>;
};

// Specialization for the simd_obj_impl type derivatives.
template <typename T, int Nsrc, int Ndst, template <typename, int> class SimdT>
struct resize_a_simd_type<SimdT<T, Nsrc>, Ndst> {
  using type = SimdT<T, Ndst>;
};

// Convenience shortcut.
template <class SimdT, int Ndst>
using resize_a_simd_type_t = typename resize_a_simd_type<SimdT, Ndst>::type;
// @}

// @{
// Converts element type of given simd type \c SimdT to
// given scalar type \c NewElemT.
template <class SimdT, typename NewElemT> struct convert_simd_elem_type;

// Specialization for the simd_obj_impl type.
template <typename OldElemT, int N, typename NewElemT,
          template <typename, int> class SimdT>
struct convert_simd_elem_type<
    simd_obj_impl<__raw_t<OldElemT>, N, SimdT<OldElemT, N>>, NewElemT> {
  using type = simd_obj_impl<__raw_t<NewElemT>, N, SimdT<NewElemT, N>>;
};

// Specialization for the simd_obj_impl type derivatives.
template <typename OldElemT, int N, typename NewElemT,
          template <typename, int> class SimdT>
struct convert_simd_elem_type<SimdT<OldElemT, N>, NewElemT> {
  using type = SimdT<NewElemT, N>;
};

// Convenience shortcut.
template <class SimdT, typename NewElemT>
using convert_simd_elem_type_t =
    typename convert_simd_elem_type<SimdT, NewElemT>::type;

// @}

// Constructs a simd type with the same template type as in \c SimdT, and
// given element type and number.
template <class SimdT, typename T, int N>
using construct_a_simd_type_t =
    convert_simd_elem_type_t<resize_a_simd_type_t<SimdT, N>, T>;

// @}

// must match simd_mask<N>::element_type
template <int N>
using simd_mask_storage_t = vector_type_t<simd_mask_elem_type, N>;

// @{
// Checks if given type is a view of any simd type (simd or simd_mask).
template <typename Ty> struct is_any_simd_view_type : std::false_type {};

template <typename BaseTy, typename RegionTy>
struct is_any_simd_view_type<simd_view<BaseTy, RegionTy>> : std::true_type {};

template <typename Ty>
static inline constexpr bool is_any_simd_view_type_v =
    is_any_simd_view_type<Ty>::value;
// @}

// @{
// Check if a type is one of internal 'simd_xxx_impl' types exposing simd-like
// interfaces and behaving like a simd object type.

template <typename Ty>
static inline constexpr bool is_simd_like_type_v =
    is_any_simd_view_type_v<Ty> || is_simd_obj_impl_derivative_v<Ty>;
// @}

// @{
// Checks if given type is a any of the user-visible simd types (simd or
// simd_mask).
template <typename Ty> struct is_simd_type : std::false_type {};
template <typename ElTy, int N>
struct is_simd_type<simd<ElTy, N>> : std::true_type {};
template <typename Ty>
static inline constexpr bool is_simd_type_v = is_simd_type<Ty>::value;

template <typename Ty> struct is_simd_mask_type : std::false_type {};
template <int N>
struct is_simd_mask_type<simd_mask_impl<simd_mask_elem_type, N>>
    : std::true_type {};
template <typename Ty>
static inline constexpr bool is_simd_mask_type_v = is_simd_mask_type<Ty>::value;
// @}

// @{
// Checks if given type is a view of the simd type.
template <typename Ty> struct is_simd_view_type_impl : std::false_type {};

template <class BaseT, class RegionT>
struct is_simd_view_type_impl<simd_view<BaseT, RegionT>>
    : std::conditional_t<is_simd_type_v<BaseT>, std::true_type,
                         std::false_type> {};

template <class Ty>
struct is_simd_view_type : is_simd_view_type_impl<remove_cvref_t<Ty>> {};

template <typename Ty>
static inline constexpr bool is_simd_view_type_v = is_simd_view_type<Ty>::value;
// @}

template <typename T>
static inline constexpr bool is_simd_or_view_type_v =
    is_simd_view_type_v<T> || is_simd_type_v<T>;

// @{
// Get the element type if it is a scalar, clang vector, simd or simd_view type.

struct cant_deduce_element_type;

template <class T, class SFINAE = void> struct element_type {
  using type = cant_deduce_element_type;
};

template <typename T>
struct element_type<T, std::enable_if_t<is_vectorizable_v<T>>> {
  using type = remove_cvref_t<T>;
};

template <typename T>
struct element_type<T, std::enable_if_t<is_simd_like_type_v<T>>> {
  using type = typename T::element_type;
};

template <typename T>
struct element_type<T, std::enable_if_t<is_clang_vector_type_v<T>>> {
  using type = typename is_clang_vector_type<T>::element_type;
};

template <typename T> using element_type_t = typename element_type<T>::type;

// Determine element type of simd_obj_impl's Derived type w/o having to have
// complete instantiation of the Derived type (is required by element_type_t,
// hence can't be used here).
template <class T> struct simd_like_obj_info {
  using element_type = T;
  static constexpr int vector_length = 0;
};

template <class T, int N> struct simd_like_obj_info<simd<T, N>> {
  using element_type = T;
  static constexpr int vector_length = N;
};

template <class T, int N> struct simd_like_obj_info<simd_mask_impl<T, N>> {
  using element_type = simd_mask_elem_type; // equals T
  static constexpr int vector_length = N;
};

template <class BaseT, class RegionT>
struct simd_like_obj_info<simd_view<BaseT, RegionT>> {
  using element_type = typename RegionT::element_type;
  static constexpr int vector_length = RegionT::length;
};

template <typename T>
using get_vector_element_type = typename simd_like_obj_info<T>::element_type;

template <typename T>
static inline constexpr int get_vector_length =
    simd_like_obj_info<T>::vector_length;

// @}

template <typename To, typename From>
std::enable_if_t<is_clang_vector_type_v<To> && is_clang_vector_type_v<From>, To>
    ESIMD_INLINE convert(From Val) {
  if constexpr (std::is_same_v<To, From>) {
    return Val;
  } else {
    return __builtin_convertvector(Val, To);
  }
}

// calculates the number of elements in "To" type
template <typename ToEltTy, typename FromEltTy, int FromN,
          typename = std::enable_if_t<is_vectorizable<ToEltTy>::value>>
struct bitcast_helper {
  static constexpr int nToElems() {
    constexpr int R1 = sizeof(ToEltTy) / sizeof(FromEltTy);
    constexpr int R2 = sizeof(FromEltTy) / sizeof(ToEltTy);
    constexpr int ToN = (R2 > 0) ? (FromN * R2) : (FromN / R1);
    return ToN;
  }
};

// Change the element type of a simd vector.
template <typename ToEltTy, typename FromEltTy, int FromN,
          typename = std::enable_if_t<is_vectorizable<ToEltTy>::value>>
ESIMD_INLINE typename std::conditional_t<
    std::is_same_v<FromEltTy, ToEltTy>, vector_type_t<FromEltTy, FromN>,
    vector_type_t<ToEltTy,
                  bitcast_helper<ToEltTy, FromEltTy, FromN>::nToElems()>>
bitcast(vector_type_t<FromEltTy, FromN> Val) {
  // Noop.
  if constexpr (std::is_same_v<FromEltTy, ToEltTy>)
    return Val;

  // Bitcast
  constexpr int ToN = bitcast_helper<ToEltTy, FromEltTy, FromN>::nToElems();
  using VTy = vector_type_t<ToEltTy, ToN>;
  return reinterpret_cast<VTy>(Val);
}

// Get computation type of a binary operator given its operand types:
// - if both types are arithmetic - return CPP's "common real type" of the
//   computation (matches C++)
// - if both types are simd types, they must be of the same length N,
//   and the returned type is simd<T, N>, where N is the "common real type" of
//   the element type of the operands (diverges from clang)
// - otherwise, one type is simd and another is arithmetic - the simd type is
//   returned (matches clang)

struct invalid_computation_type;

template <class T1, class T2, class SFINAE = void> struct computation_type {
  using type = invalid_computation_type;
};

template <class T1, class T2>
struct computation_type<T1, T2,
                        std::enable_if_t<is_valid_simd_elem_type_v<T1> &&
                                         is_valid_simd_elem_type_v<T2>>> {
private:
  template <class T> using tr = element_type_traits<T>;
  template <class T>
  using native_t =
      std::conditional_t<tr<T>::use_native_cpp_ops, typename tr<T>::RawT,
                         typename tr<T>::EnclosingCppT>;
  static constexpr bool is_wr1 = is_wrapper_elem_type_v<T1>;
  static constexpr bool is_wr2 = is_wrapper_elem_type_v<T2>;
  static constexpr bool is_fp1 = is_generic_floating_point_v<T1>;
  static constexpr bool is_fp2 = is_generic_floating_point_v<T2>;

public:
  using type = std::conditional_t<
      !is_wr1 && !is_wr2,
      // T1 and T2 are both std C++ types - use std C++ type promotion
      decltype(std::declval<T1>() + std::declval<T2>()),
      std::conditional_t<
          std::is_same_v<T1, T2>,
          // Types are the same wrapper type - return any
          T1,
          std::conditional_t<is_fp1 != is_fp2,
                             // One of the types is floating-point - return it
                             // (e.g. computation_type<int, sycl::half> will
                             // yield sycl::half)
                             std::conditional_t<is_fp1, T1, T2>,
                             // both are either floating point or integral -
                             // return result of C++ promotion of the native
                             // types
                             decltype(std::declval<native_t<T1>>() +
                                      std::declval<native_t<T2>>())>>>;
};

template <class T1, class T2>
struct computation_type<
    T1, T2,
    std::enable_if_t<is_simd_like_type_v<T1> || is_simd_like_type_v<T2>>> {
private:
  using Ty1 = element_type_t<T1>;
  using Ty2 = element_type_t<T2>;
  using EltTy = typename computation_type<Ty1, Ty2>::type;

  static constexpr int N1 = [] {
    if constexpr (is_simd_like_type_v<T1>) {
      return T1::length;
    } else {
      return 0;
    }
  }();
  static constexpr int N2 = [] {
    if constexpr (is_simd_like_type_v<T2>) {
      return T2::length;
    } else {
      return 0;
    }
  }();
  static_assert((N1 == N2) || (N1 == 0) || (N2 == 0), "size mismatch");
  static constexpr int N = N1 ? N1 : N2;

public:
  using type = simd<EltTy, N>;
};

template <class T1, class T2 = T1>
using computation_type_t =
    typename computation_type<remove_cvref_t<T1>, remove_cvref_t<T2>>::type;

} // namespace detail

/// @endcond ESIMD_DETAIL

// Alias for backward compatibility.
template <int N> using mask_type_t = detail::simd_mask_storage_t<N>;

} // namespace ext::intel::esimd
} // namespace _V1
} // namespace sycl
