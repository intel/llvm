//==-------------- region.hpp - DPC++ Explicit SIMD API --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Region type to implement the Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#include <CL/sycl/detail/defines.hpp>
#include <cstdint>
#include <type_traits>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace esimd {

/// @cond ESIMD_DETAIL
// TODO move to detail?

// The common base type of region types.
template <bool Is2D, typename T, int SizeY, int StrideY, int SizeX, int StrideX>
struct region_base {
  using element_type = T;
  static constexpr int length = SizeX * SizeY;
  static constexpr int Is_2D = Is2D;
  static constexpr int Size_x = SizeX;
  static constexpr int Stride_x = StrideX;
  static constexpr int Size_y = SizeY;
  static constexpr int Stride_y = StrideY;
  static constexpr int Size_in_bytes = sizeof(T) * length;

  static_assert(Size_x > 0 && Stride_x >= 0, "illegal region in x-dimension");
  static_assert(Size_y > 0 && Stride_y >= 0, "illegal region in y-dimension");

  uint16_t M_offset_y;
  uint16_t M_offset_x;

  explicit region_base() : M_offset_y(0), M_offset_x(0) {}

  explicit region_base(uint16_t OffsetX) : M_offset_y(0), M_offset_x(OffsetX) {}

  explicit region_base(uint16_t OffsetY, uint16_t OffsetX)
      : M_offset_y(OffsetY), M_offset_x(OffsetX) {}
};

// A basic 1D region type.
template <typename T, int Size, int Stride>
using region1d_t = region_base<false, T, 1, 1, Size, Stride>;

// A basic 2D region type.
template <typename T, int SizeY, int StrideY, int SizeX, int StrideX>
using region2d_t = region_base<true, T, SizeY, StrideY, SizeX, StrideX>;

// A region with a single element.
template <typename T>
using region1d_scalar_t =
    region_base<false, T, 1 /*SizeY*/, 1 /*StrideY*/, 1, 1>;

// simd_view forward declaration.
template <typename BaseTy, typename RegionTy> class simd_view;

// Compute the region type of a simd_view type.
//
// A region type could be either
// - region1d_t
// - region2d_t
// - a pair (top_region_type, base_region_type)
//
// This is a recursive definition to capture the following rvalue region stack:
//
// simd<int 16> v;
// v.bit_cast_view<int, 4, 4>().select<1, 0, 4, 1>(0, 0).bit_cast_view<short>()
// = 0;
//
// The LHS will be represented as a rvalue
//
// simd_view({v, { region1d_t<short, 8, 1>(0, 0),
//              { region2d_t<int, 1, 1, 4, 1>(0, 0),
//                region2d_t<int, 4, 1, 4, 1>(0, 0)
//              }}})
//
template <typename Ty> struct shape_type {
  using element_type = Ty;
  using type = void;
};

template <typename Ty, int Size, int Stride>
struct shape_type<region1d_t<Ty, Size, Stride>> {
  using element_type = Ty;
  using type = region1d_t<Ty, Size, Stride>;
  static inline constexpr int length = type::length;
};

template <typename Ty> struct shape_type<region1d_scalar_t<Ty>> {
  using element_type = Ty;
  using type = region1d_t<Ty, 1, 1>;
  static inline constexpr int length = type::length;
};

template <typename Ty, int SizeY, int StrideY, int SizeX, int StrideX>
struct shape_type<region2d_t<Ty, SizeY, StrideY, SizeX, StrideX>> {
  using element_type = Ty;
  using type = region2d_t<Ty, SizeY, StrideY, SizeX, StrideX>;
  static inline constexpr int length = type::length;
};

// Forward the shape computation on the top region type.
template <typename TopRegionTy, typename BaseRegionTy>
struct shape_type<std::pair<TopRegionTy, BaseRegionTy>>
    : public shape_type<TopRegionTy> {};

// Forward the shape computation on the region type.
template <typename BaseTy, typename RegionTy>
struct shape_type<simd_view<BaseTy, RegionTy>> : public shape_type<RegionTy> {};

// Utility functions to access region components.
template <typename T> T getTopRegion(T Reg) { return Reg; }
template <typename T, typename U> T getTopRegion(std::pair<T, U> Reg) {
  return Reg.first;
}

template <typename T> T getBaseRegion(T Reg) { return Reg; }
template <typename T, typename U> T getBaseRegion(std::pair<T, U> Reg) {
  return Reg.second;
}

/// @endcond ESIMD_DETAIL

} // namespace esimd
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

/// @endcond ESIMD_DETAIL
