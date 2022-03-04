//==-------------- types.hpp - DPC++ Explicit SIMD API ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Meta-functions to compute compile-time element type of a simd_view resulting
// from format operations.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#include <sycl/ext/intel/esimd/detail/types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace __ESIMD_DNS {

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

} // namespace __ESIMD_DNS
} // __SYCL_INLINE_NAMESPACE(cl)

/// @endcond ESIMD_DETAIL
