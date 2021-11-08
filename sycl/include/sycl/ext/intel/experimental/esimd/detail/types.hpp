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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

// simd and simd_view_impl forward declarations
template <typename Ty, int N> class simd;
template <typename BaseTy, typename RegionTy> class simd_view;

namespace detail {

// forward declarations of major internal simd classes
template <typename Ty, int N> class simd_mask_impl;
template <typename ElT, int N, class Derived, class SFINAE = void>
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

// @{
// Checks if given type T derives from simd_obj_impl or is equal to it.
template <typename T>
struct is_simd_obj_impl_derivative : public std::false_type {
  using element_type = invalid_element_type;
};

// Specialization for the simd_obj_impl type itself.
template <typename ElT, int N, class Derived>
struct is_simd_obj_impl_derivative<simd_obj_impl<ElT, N, Derived>>
    : public std::true_type {
  using element_type = ElT;
};

// Specialization for all other types.
template <typename ElT, int N, template <typename, int> class Derived>
struct is_simd_obj_impl_derivative<Derived<ElT, N>>
    : public std::conditional_t<
          std::is_base_of_v<simd_obj_impl<ElT, N, Derived<ElT, N>>,
                            Derived<ElT, N>>,
          std::true_type, std::false_type> {
  using element_type = std::conditional_t<
      std::is_base_of_v<simd_obj_impl<ElT, N, Derived<ElT, N>>,
                        Derived<ElT, N>>,
      ElT, void>;
};

// Convenience shortcut.
template <typename T>
inline constexpr bool is_simd_obj_impl_derivative_v =
    is_simd_obj_impl_derivative<T>::value;
// @}

// @{
// "Resizes" given simd type \c T to given number of elements \c N.
template <class SimdT, int Ndst> struct resize_a_simd_type;

// Specialization for the simd_obj_impl type.
template <typename ElT, int Nsrc, int Ndst,
          template <typename, int> class SimdT>
struct resize_a_simd_type<simd_obj_impl<ElT, Nsrc, SimdT<ElT, Nsrc>>, Ndst> {
  using type = simd_obj_impl<ElT, Ndst, SimdT<ElT, Ndst>>;
};

// Specialization for the simd_obj_impl type derivatives.
template <typename ElT, int Nsrc, int Ndst,
          template <typename, int> class SimdT>
struct resize_a_simd_type<SimdT<ElT, Nsrc>, Ndst> {
  using type = SimdT<ElT, Ndst>;
};

// Convenience shortcut.
template <class SimdT, int Ndst>
using resize_a_simd_type_t = typename resize_a_simd_type<SimdT, Ndst>::type;
// @}

// @{
// Converts element type of given simd type \c SimdT to
// given scalar type \c DstElemT.
template <class SimdT, typename DstElemT> struct convert_simd_elem_type;

// Specialization for the simd_obj_impl type.
template <typename SrcElemT, int N, typename DstElemT,
          template <typename, int> class SimdT>
struct convert_simd_elem_type<simd_obj_impl<SrcElemT, N, SimdT<SrcElemT, N>>,
                              DstElemT> {
  using type = simd_obj_impl<DstElemT, N, SimdT<DstElemT, N>>;
};

// Specialization for the simd_obj_impl type derivatives.
template <typename SrcElemT, int N, typename DstElemT,
          template <typename, int> class SimdT>
struct convert_simd_elem_type<SimdT<SrcElemT, N>, DstElemT> {
  using type = SimdT<DstElemT, N>;
};

// Convenience shortcut.
template <class SimdT, typename DstElemT>
using convert_simd_elem_type_t =
    typename convert_simd_elem_type<SimdT, DstElemT>::type;

// @}

// Constructs a simd type with the same template type as in \c SimdT, and
// given element type and number.
template <class SimdT, typename ElT, int N>
using construct_a_simd_type_t =
    convert_simd_elem_type_t<resize_a_simd_type_t<SimdT, N>, ElT>;

// @}

namespace csd = cl::sycl::detail;
using half = cl::sycl::detail::half_impl::StorageT;

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

// vector_type, using clang vector type extension.
template <typename Ty, int N> struct vector_type {
  static_assert(!std::is_const<Ty>::value, "const element type not supported");
  static_assert(is_vectorizable_v<Ty>, "element type not supported");
  static_assert(N > 0, "zero-element vector not supported");

  static constexpr int length = N;
  using type = Ty __attribute__((ext_vector_type(N)));
};

template <typename Ty, int N>
using vector_type_t = typename vector_type<Ty, N>::type;

// must match simd_mask<N>::element_type
template <int N>
using simd_mask_storage_t = vector_type_t<simd_mask_elem_type, N>;

// Compute the simd_view type of a 1D format operation.
template <typename BaseTy, typename EltTy> struct compute_format_type;

template <typename Ty, int N, typename EltTy> struct compute_format_type_impl {
  static constexpr int Size = sizeof(Ty) * N / sizeof(EltTy);
  static constexpr int Stride = 1;
  using type = region1d_t<EltTy, Size, Stride>;
};

template <typename Ty, int N, typename EltTy,
          template <typename, int> class SimdT>
struct compute_format_type<SimdT<Ty, N>, EltTy>
    : compute_format_type_impl<Ty, N, EltTy> {};

template <typename Ty, int N, typename EltTy, class SimdT>
struct compute_format_type<simd_obj_impl<Ty, N, SimdT>, EltTy>
    : compute_format_type_impl<Ty, N, EltTy> {};

template <typename BaseTy, typename RegionTy, typename EltTy>
struct compute_format_type<simd_view<BaseTy, RegionTy>, EltTy> {
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int Size = ShapeTy::Size_in_bytes / sizeof(EltTy);
  static constexpr int Stride = 1;
  using type = region1d_t<EltTy, Size, Stride>;
};

template <typename Ty, typename EltTy>
using compute_format_type_t = typename compute_format_type<Ty, EltTy>::type;

// Compute the simd_view type of a 2D format operation.
template <typename BaseTy, typename EltTy, int Height, int Width>
struct compute_format_type_2d;

template <typename Ty, int N, typename EltTy, int Height, int Width>
struct compute_format_type_2d_impl {
  static constexpr int Prod = sizeof(Ty) * N / sizeof(EltTy);
  static_assert(Prod == Width * Height, "size mismatch");

  static constexpr int SizeX = Width;
  static constexpr int StrideX = 1;
  static constexpr int SizeY = Height;
  static constexpr int StrideY = 1;
  using type = region2d_t<EltTy, SizeY, StrideY, SizeX, StrideX>;
};

template <typename Ty, int N, typename EltTy, int Height, int Width,
          template <typename, int> class SimdT>
struct compute_format_type_2d<SimdT<Ty, N>, EltTy, Height, Width>
    : compute_format_type_2d_impl<Ty, N, EltTy, Height, Width> {};

template <typename Ty, int N, typename EltTy, int Height, int Width,
          class SimdT>
struct compute_format_type_2d<simd_obj_impl<Ty, N, SimdT>, EltTy, Height, Width>
    : compute_format_type_2d_impl<Ty, N, EltTy, Height, Width> {};

template <typename BaseTy, typename RegionTy, typename EltTy, int Height,
          int Width>
struct compute_format_type_2d<simd_view<BaseTy, RegionTy>, EltTy, Height,
                              Width> {
  using ShapeTy = typename shape_type<RegionTy>::type;
  static constexpr int Prod = ShapeTy::Size_in_bytes / sizeof(EltTy);
  static_assert(Prod == Width * Height, "size mismatch");

  static constexpr int SizeX = Width;
  static constexpr int StrideX = 1;
  static constexpr int SizeY = Height;
  static constexpr int StrideY = 1;
  using type = region2d_t<EltTy, SizeY, StrideY, SizeX, StrideX>;
};

template <typename Ty, typename EltTy, int Height, int Width>
using compute_format_type_2d_t =
    typename compute_format_type_2d<Ty, EltTy, Height, Width>::type;

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

// @}

// @{
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
struct computation_type<
    T1, T2, std::enable_if_t<is_vectorizable_v<T1> && is_vectorizable_v<T2>>> {
  using type = decltype(std::declval<T1>() + std::declval<T2>());
};

template <class T1, class T2>
struct computation_type<
    T1, T2,
    std::enable_if_t<is_simd_like_type_v<T1> && is_simd_like_type_v<T2>>> {
private:
  using Ty1 = typename element_type<T1>::type;
  using Ty2 = typename element_type<T2>::type;
  using EltTy = typename computation_type<Ty1, Ty2>::type;
  static constexpr int N1 = T1::length;
  static constexpr int N2 = T2::length;
  static_assert(N1 == N2, "size mismatch");

public:
  using type = simd<EltTy, N1>;
};

template <class T1, class T2 = T1>
using computation_type_t =
    typename computation_type<remove_cvref_t<T1>, remove_cvref_t<T2>>::type;

// @}

template <typename To, typename From>
std::enable_if_t<is_clang_vector_type_v<To> && is_clang_vector_type_v<From>, To>
convert(From Val) {
  return __builtin_convertvector(Val, To);
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

inline std::ostream &operator<<(std::ostream &O, half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

inline std::istream &operator>>(std::istream &I, half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

} // namespace detail

// Alias for backward compatibility.
template <int N> using mask_type_t = detail::simd_mask_storage_t<N>;

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
