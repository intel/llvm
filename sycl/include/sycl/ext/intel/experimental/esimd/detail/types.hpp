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

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/stl_type_traits.hpp> // to define C++14,17 extensions
#include <CL/sycl/half_type.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/region.hpp>

#if defined(__ESIMD_DBG_HOST) && !defined(__SYCL_DEVICE_ONLY__)
#define __esimd_dbg_print(a) std::cout << ">>> " << #a << "\n"
#else
#define __esimd_dbg_print(a)
#endif // defined(__ESIMD_DBG_HOST) && !defined(__SYCL_DEVICE_ONLY__)

#include <cstdint>

#define __SEIEED sycl::ext::intel::experimental::esimd::detail
#define __SEIEE sycl::ext::intel::experimental::esimd
#define __SEIEEED sycl::ext::intel::experimental::esimd::emu::detail

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

// simd and simd_view_impl forward declarations
template <typename Ty, int N> class simd;
template <typename BaseTy, typename RegionTy> class simd_view;

/// @cond ESIMD_DETAIL

namespace detail {

namespace csd = cl::sycl::detail;

template <int N>
using uint_type_t = std::conditional_t<
    N == 1, uint8_t,
    std::conditional_t<
        N == 2, uint16_t,
        std::conditional_t<N == 4, uint32_t,
                           std::conditional_t<N == 8, uint64_t, void>>>>;

// forward declarations of major internal simd classes
template <typename Ty, int N> class simd_mask_impl;
template <typename RawT, int N, class Derived, class SFINAE = void>
class simd_obj_impl;

// @{
// Helpers for major simd classes, which don't require their definitions to
// compile. Error checking/SFINAE is not used as these are only used internally.

using simd_mask_elem_type = unsigned short;
template <int N> using simd_mask_type = simd_mask_impl<simd_mask_elem_type, N>;

// @{
// Checks if given type T is a raw clang vector type, plus provides some info
// about it if it is.

struct invalid_element_type;

template <class T> struct is_clang_vector_type : std::false_type {
  static inline constexpr int length = 0;
  using element_type = invalid_element_type;
};

template <class T, int N>
struct is_clang_vector_type<T __attribute__((ext_vector_type(N)))>
    : std::true_type {
  static inline constexpr int length = N;
  using element_type = T;
};
template <class T>
static inline constexpr bool is_clang_vector_type_v =
    is_clang_vector_type<T>::value;

// @}

template <typename T>
using remove_cvref_t = csd::remove_cv_t<csd::remove_reference_t<T>>;

// is_esimd_arithmetic_type
template <class...> struct make_esimd_void { using type = void; };
template <class... Tys>
using __esimd_void_t = typename make_esimd_void<Tys...>::type;

template <class Ty, class = void>
struct is_esimd_arithmetic_type : std::false_type {};

template <class Ty>
struct is_esimd_arithmetic_type<
    Ty, __esimd_void_t<std::enable_if_t<std::is_arithmetic_v<Ty>>,
                       decltype(std::declval<Ty>() + std::declval<Ty>()),
                       decltype(std::declval<Ty>() - std::declval<Ty>()),
                       decltype(std::declval<Ty>() * std::declval<Ty>()),
                       decltype(std::declval<Ty>() / std::declval<Ty>())>>
    : std::true_type {};

template <typename Ty>
static inline constexpr bool is_esimd_arithmetic_type_v =
    is_esimd_arithmetic_type<Ty>::value;

// is_vectorizable_type
template <typename Ty>
struct is_vectorizable : std::conditional_t<is_esimd_arithmetic_type_v<Ty>,
                                            std::true_type, std::false_type> {};

template <typename Ty>
static inline constexpr bool is_vectorizable_v = is_vectorizable<Ty>::value;

template <typename T>
static inline constexpr bool is_esimd_scalar_v =
    cl::sycl::detail::is_arithmetic<T>::value;

template <typename T>
using is_esimd_scalar = typename std::bool_constant<is_esimd_scalar_v<T>>;

// raw_vector_type, using clang vector type extension.
template <typename Ty, int N> struct raw_vector_type {
  static_assert(!std::is_const<Ty>::value, "const element type not supported");
  static_assert(is_vectorizable_v<Ty>, "element type not supported");
  static_assert(N > 0, "zero-element vector not supported");

  static constexpr int length = N;
  using type = Ty __attribute__((ext_vector_type(N)));
};

template <typename Ty, int N>
using vector_type_t = typename raw_vector_type<Ty, N>::type;

// @{
// Checks if given type T derives from simd_obj_impl or is equal to it.
template <typename T>
struct is_simd_obj_impl_derivative : public std::false_type {};

// Specialization for the simd_obj_impl type itself.
template <typename RawT, int N, class Derived>
struct is_simd_obj_impl_derivative<simd_obj_impl<RawT, N, Derived>>
    : public std::true_type {};

template <class T, class SFINAE = void> struct element_type_traits;
template <class T>
using __raw_t = typename __SEIEED::element_type_traits<T>::RawT;
template <class T>
using __cpp_t = typename __SEIEED::element_type_traits<T>::EnclosingCppT;

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
  static inline constexpr int vector_length = 0;
};

template <class T, int N> struct simd_like_obj_info<simd<T, N>> {
  using element_type = T;
  static inline constexpr int vector_length = N;
};

template <class T, int N> struct simd_like_obj_info<simd_mask_impl<T, N>> {
  using element_type = simd_mask_elem_type; // equals T
  static inline constexpr int vector_length = N;
};

template <class BaseT, class RegionT>
struct simd_like_obj_info<simd_view<BaseT, RegionT>> {
  using element_type = typename RegionT::element_type;
  static inline constexpr int vector_length = RegionT::length;
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

/// Base case for checking if a type U is one of the types.
template <typename U> constexpr bool is_type() { return false; }

template <typename U, typename T, typename... Ts> constexpr bool is_type() {
  using UU = typename csd::remove_const_t<U>;
  using TT = typename csd::remove_const_t<T>;
  return std::is_same<UU, TT>::value || is_type<UU, Ts...>();
}

// calculates the number of elements in "To" type
template <typename ToEltTy, typename FromEltTy, int FromN,
          typename = csd::enable_if_t<is_vectorizable<ToEltTy>::value>>
struct bitcast_helper {
  static inline constexpr int nToElems() {
    constexpr int R1 = sizeof(ToEltTy) / sizeof(FromEltTy);
    constexpr int R2 = sizeof(FromEltTy) / sizeof(ToEltTy);
    constexpr int ToN = (R2 > 0) ? (FromN * R2) : (FromN / R1);
    return ToN;
  }
};

// Change the element type of a simd vector.
template <typename ToEltTy, typename FromEltTy, int FromN,
          typename = csd::enable_if_t<is_vectorizable<ToEltTy>::value>>
ESIMD_INLINE typename csd::conditional_t<
    std::is_same<FromEltTy, ToEltTy>::value, vector_type_t<FromEltTy, FromN>,
    vector_type_t<ToEltTy,
                  bitcast_helper<ToEltTy, FromEltTy, FromN>::nToElems()>>
bitcast(vector_type_t<FromEltTy, FromN> Val) {
  // Noop.
  if constexpr (std::is_same<FromEltTy, ToEltTy>::value)
    return Val;

  // Bitcast
  constexpr int ToN = bitcast_helper<ToEltTy, FromEltTy, FromN>::nToElems();
  using VTy = vector_type_t<ToEltTy, ToN>;
  return reinterpret_cast<VTy>(Val);
}

} // namespace detail

/// @endcond ESIMD_DETAIL

// Alias for backward compatibility.
template <int N> using mask_type_t = detail::simd_mask_storage_t<N>;

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
