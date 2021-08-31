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

#include <cstdint>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

// simd and simd_view forward declarations
template <typename Ty, int N> class simd;
template <typename BaseTy, typename RegionTy> class simd_view;

namespace detail {

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
    Ty, __esimd_void_t<decltype(std::declval<Ty>() + std::declval<Ty>()),
                       decltype(std::declval<Ty>() - std::declval<Ty>()),
                       decltype(std::declval<Ty>() * std::declval<Ty>()),
                       decltype(std::declval<Ty>() / std::declval<Ty>())>>
    : std::true_type {};

// is_vectorizable_type
template <typename Ty>
struct is_vectorizable : public is_esimd_arithmetic_type<Ty> {};

template <> struct is_vectorizable<bool> : public std::false_type {};

template <typename Ty>
struct is_vectorizable_v
    : std::integral_constant<bool, is_vectorizable<Ty>::value> {};

// vector_type, using clang vector type extension.
template <typename Ty, int N> struct vector_type {
  static_assert(!std::is_const<Ty>::value, "const element type not supported");
  static_assert(is_vectorizable_v<Ty>::value, "element type not supported");
  static_assert(N > 0, "zero-element vector not supported");

  static constexpr int length = N;
  using type = Ty __attribute__((ext_vector_type(N)));
};

template <typename Ty, int N>
using vector_type_t = typename vector_type<Ty, N>::type;

// Compute the simd_view type of a 1D format operation.
template <typename BaseTy, typename EltTy> struct compute_format_type;

template <typename Ty, int N, typename EltTy>
struct compute_format_type<simd<Ty, N>, EltTy> {
  static constexpr int Size = sizeof(Ty) * N / sizeof(EltTy);
  static constexpr int Stride = 1;
  using type = region1d_t<EltTy, Size, Stride>;
};

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
struct compute_format_type_2d<simd<Ty, N>, EltTy, Height, Width> {
  static constexpr int Prod = sizeof(Ty) * N / sizeof(EltTy);
  static_assert(Prod == Width * Height, "size mismatch");

  static constexpr int SizeX = Width;
  static constexpr int StrideX = 1;
  static constexpr int SizeY = Height;
  static constexpr int StrideY = 1;
  using type = region2d_t<EltTy, SizeY, StrideY, SizeX, StrideX>;
};

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

// Check if a type is simd_view type
template <typename Ty> struct is_simd_view_type : std::false_type {};

template <typename BaseTy, typename RegionTy>
struct is_simd_view_type<simd_view<BaseTy, RegionTy>> : std::true_type {};

template <typename Ty>
struct is_simd_view_v
    : std::integral_constant<bool,
                             is_simd_view_type<remove_cvref_t<Ty>>::value> {};

// Check if a type is simd or simd_view type
template <typename Ty> struct is_simd_type : std::false_type {};

template <typename Ty, int N>
struct is_simd_type<simd<Ty, N>> : std::true_type {};

template <typename BaseTy, typename RegionTy>
struct is_simd_type<simd_view<BaseTy, RegionTy>> : std::true_type {};

template <typename Ty>
struct is_simd_v
    : std::integral_constant<bool, is_simd_type<remove_cvref_t<Ty>>::value> {};

// Get the element type if it is a simd or simd_view type.
template <typename Ty> struct element_type { using type = remove_cvref_t<Ty>; };
template <typename Ty, int N> struct element_type<simd<Ty, N>> {
  using type = Ty;
};
template <typename BaseTy, typename RegionTy>
struct element_type<simd_view<BaseTy, RegionTy>> {
  using type = typename RegionTy::element_type;
};

// Get the common type of a binary operator.
template <typename T1, typename T2,
          typename =
              csd::enable_if_t<is_simd_v<T1>::value && is_simd_v<T2>::value>>
struct common_type {
private:
  using Ty1 = typename element_type<T1>::type;
  using Ty2 = typename element_type<T2>::type;
  using EltTy = decltype(Ty1() + Ty2());
  static constexpr int N1 = T1::length;
  static constexpr int N2 = T2::length;
  static_assert(N1 == N2, "size mismatch");

public:
  using type = simd<EltTy, N1>;
};

template <typename T1, typename T2 = T1>
using compute_type_t =
    typename common_type<remove_cvref_t<T1>, remove_cvref_t<T2>>::type;

template <typename To, typename From> To convert(From Val) {
  return __builtin_convertvector(Val, To);
}

/// Get the computation type.
template <typename T1, typename T2> struct computation_type {
  // Currently only arithmetic operations are needed.
  typedef decltype(T1() + T2()) type;
};

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

// TODO @rolandschulz on May 21
// {quote}
// - The mask should also be a wrapper around the clang - vector type rather
//   than the clang - vector type itself.
// - The internal storage should be implementation defined.uint16_t is a bad
//   choice for some HW.Nor is it how clang - vector types works(using the same
//   size int as the corresponding vector type used for comparison(e.g. long for
//   double and int for float)).
template <int N>
using mask_type_t = typename detail::vector_type<uint16_t, N>::type;

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
