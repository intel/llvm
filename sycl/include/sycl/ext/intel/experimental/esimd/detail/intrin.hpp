//==------------ intrin.hpp - DPC++ Explicit SIMD API   --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares Explicit SIMD intrinsics used to implement working with
// the SIMD classes objects.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>

#include <assert.h>
#include <cstdint>

// \brief __esimd_rdregion: region access intrinsic.
//
// @param T the element data type, one of i8, i16, i32, i64, half, float,
// double. In particular bool (i1) and pointer types are not allowed.
//
// @param N the input vector size.
//
// @param M the return vector size.
//
// @param VStride the vertical stride in elements between rows.
//
// @param Width the size or each row, non-zero and even divides `M`.
//
// @param Stride horizontal stride in elements within each row.
//
// @param ParentWidth the width of the input vector when viewed as a 2D
// matrix. Ignored if offset is a constant.
//
// @param Input the input vector
//
// @param Offset the starting offset in bytes.
//
// @return the region extracted.
//
// This intrinsic computes a vector Result:
//
// \code{.cpp}
// uint16_t EltOffset = Offset / sizeof(T);
// assert(Offset % sizeof(T) == 0);
//
// int NumRows = M / Width;
// assert(M % Width == 0);
//
// int Index = 0;
// for (int i = 0; i < NumRows; ++i) {
//   for (int j = 0; j < Width; ++j) {
//     Result[Index++] = Input[i * VStride +  j * Stride +
//     EltOffset];
//   }
// }
// \endcode
//
template <typename T, int N, int M, int VStride, int Width, int Stride,
          int ParentWidth = 0>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, M>
__esimd_rdregion(__SEIEED::vector_type_t<T, N> Input, uint16_t Offset);

template <typename T, int N, int M, int ParentWidth = 0>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, M>
__esimd_rdindirect(__SEIEED::vector_type_t<T, N> Input,
                   __SEIEED::vector_type_t<uint16_t, M> Offset);

// __esimd_wrregion returns the updated vector with the region updated.
//
// @param T the element data type, one of i8, i16, i32, i64, half, float,
// double. In particular bool (i1) and pointer types are not allowed.
//
// @param N the return vector size.
//
// @param M the vector size to write.
//
// @param VStride the vertical stride in elements between rows.
//
// @param Width the size or each row, non-zero and even divides `M`.
//
// @param Stride horizontal stride in elements within each row.
//
// @param ParentWidth the width of the input vector when viewed as a 2D
// matrix. Ignored if offset is a constant.
//
// @param OldVal the vector to write region into.
//
// @param NewVal the vector to write.
//
// @param Offset the starting offset in bytes.
//
// @return the updated vector with the region modifided.
//
// This intrinsic computes a vector Result:
//
// \code{.cpp}
// uint16_t EltOffset = Offset / sizeof(T);
// assert(Offset % sizeof(T) == 0);
//
// int NumRows = M / Width;
// assert(M % Width == 0);
//
// Result = OldValue;
// int Index = 0;
// for (int i = 0; i < NumRows; ++i) {
//   for (int j = 0; j < Width; ++j) {
//       if (Mask[Index])
//           Result[i * VStride +  j * Stride + EltOffset] =
//           NewVal[Index];
//       ++Index;
//   }
// }
// \endcode
//
template <typename T, int N, int M, int VStride, int Width, int Stride,
          int ParentWidth = 0>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, N>
__esimd_wrregion(__SEIEED::vector_type_t<T, N> OldVal,
                 __SEIEED::vector_type_t<T, M> NewVal, uint16_t Offset,
                 __SEIEED::simd_mask_storage_t<M> Mask = 1);

template <typename T, int N, int M, int ParentWidth = 0>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, N>
__esimd_wrindirect(__SEIEED::vector_type_t<T, N> OldVal,
                   __SEIEED::vector_type_t<T, M> NewVal,
                   __SEIEED::vector_type_t<uint16_t, M> Offset,
                   __SEIEED::simd_mask_storage_t<M> Mask = 1);

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

template <class T> using __st = __raw_t<T>;

/// read from a basic region of a vector, return a vector
template <typename BT, int BN, typename RTy>
__SEIEED::vector_type_t<__st<typename RTy::element_type>, RTy::length>
    ESIMD_INLINE readRegion(const __SEIEED::vector_type_t<__st<BT>, BN> &Base,
                            RTy Region) {
  using ElemTy = __st<typename RTy::element_type>;
  auto Base1 = bitcast<ElemTy, __st<BT>, BN>(Base);
  constexpr int Bytes = BN * sizeof(BT);
  if constexpr (Bytes == RTy::Size_in_bytes)
    // This is a no-op format.
    return Base1;
  else {
    static_assert(!RTy::Is_2D);
    constexpr int N = Bytes / sizeof(ElemTy);
    // Access the region information.
    constexpr int M = RTy::Size_x;
    constexpr int Stride = RTy::Stride_x;
    int16_t Offset = static_cast<int16_t>(Region.M_offset_x * sizeof(ElemTy));
    // read-region
    return __esimd_rdregion<ElemTy, N, M, /*VS*/ 0, M, Stride>(Base1, Offset);
  }
}

/// read from a nested region of a vector, return a vector
template <typename BT, int BN, typename T, typename U>
ESIMD_INLINE __SEIEED::vector_type_t<__st<typename T::element_type>, T::length>
readRegion(const __SEIEED::vector_type_t<__st<BT>, BN> &Base,
           std::pair<T, U> Region) {
  // parent-region type
  using PaTy = typename shape_type<U>::type;
  constexpr int BN1 = PaTy::length;
  using BT1 = typename PaTy::element_type;
  using ElemTy = __st<typename T::element_type>;
  // Recursively read the base
  auto Base1 = readRegion<BT, BN>(Base, Region.second);
  if constexpr (!T::Is_2D || BN1 * sizeof(BT1) == T::Size_in_bytes)
    // 1-D region or format
    return readRegion<BT1, BN1>(Base1, Region.first);
  else {
    static_assert(T::Is_2D);
    static_assert(std::is_same<ElemTy, __st<BT1>>::value);
    // To read a 2D region, we need the parent region
    // Read full rows with non-trivial vertical and horizontal stride = 1.
    constexpr int M = T::Size_y * PaTy::Size_x;
    constexpr int VS = T::Stride_y * PaTy::Size_x;
    constexpr int W = PaTy::Size_x;
    constexpr int HS = 1;
    constexpr int ParentWidth = PaTy::Size_x;
    uint16_t Offset = static_cast<uint16_t>(Region.first.M_offset_y *
                                            PaTy::Size_x * sizeof(ElemTy));

    auto R =
        __esimd_rdregion<ElemTy, BN1, M, VS, W, HS, ParentWidth>(Base1, Offset);

    // Read columns with non-trivial horizontal stride.
    constexpr int N1 = M;
    constexpr int M1 = T::length;
    constexpr int VS1 = PaTy::Size_x;
    constexpr int W1 = T::Size_x;
    constexpr int HS1 = T::Stride_x;
    uint16_t Offset1 =
        static_cast<uint16_t>(Region.first.M_offset_x * sizeof(ElemTy));

    return __esimd_rdregion<ElemTy, N1, M1, VS1, W1, HS1, ParentWidth>(R,
                                                                       Offset1);
  }
}

} // namespace detail

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

// vload
//
// map to the backend vload intrinsic, used by compiler to control
// optimization on simd object
//
template <typename T, int N>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, N>
__esimd_vload(const __SEIEED::vector_type_t<T, N> *ptr);

// vstore
//
// map to the backend vstore intrinsic, used by compiler to control
// optimization on simd object
template <typename T, int N>
__ESIMD_INTRIN void __esimd_vstore(__SEIEED::vector_type_t<T, N> *ptr,
                                   __SEIEED::vector_type_t<T, N> vals);

template <typename T, int N>
__ESIMD_INTRIN uint16_t __esimd_any(__SEIEED::vector_type_t<T, N> src)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  for (unsigned int i = 0; i != N; i++) {
    if (src[i] != 0)
      return 1;
  }
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

template <typename T, int N>
__ESIMD_INTRIN uint16_t __esimd_all(__SEIEED::vector_type_t<T, N> src)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  for (unsigned int i = 0; i != N; i++) {
    if (src[i] == 0)
      return 0;
  }
  return 1;
}
#endif // __SYCL_DEVICE_ONLY__

#ifndef __SYCL_DEVICE_ONLY__

// Implementations of ESIMD intrinsics for the SYCL host device
template <typename T, int N, int M, int VStride, int Width, int Stride,
          int ParentWidth>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, M>
__esimd_rdregion(__SEIEED::vector_type_t<T, N> Input, uint16_t Offset) {
  uint16_t EltOffset = Offset / sizeof(T);
  assert(Offset % sizeof(T) == 0);

  int NumRows = M / Width;
  assert(M % Width == 0);

  __SEIEED::vector_type_t<T, M> Result;
  int Index = 0;
  for (int i = 0; i < NumRows; ++i) {
    for (int j = 0; j < Width; ++j) {
      Result[Index++] = Input[i * VStride + j * Stride + EltOffset];
    }
  }
  return Result;
}

template <typename T, int N, int M, int ParentWidth>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, M>
__esimd_rdindirect(__SEIEED::vector_type_t<T, N> Input,
                   __SEIEED::vector_type_t<uint16_t, M> Offset) {
  __SEIEED::vector_type_t<T, M> Result;
  for (int i = 0; i < M; ++i) {
    uint16_t EltOffset = Offset[i] / sizeof(T);
    assert(Offset[i] % sizeof(T) == 0);
    assert(EltOffset < N);
    Result[i] = Input[EltOffset];
  }
  return Result;
}

template <typename T, int N, int M, int VStride, int Width, int Stride,
          int ParentWidth>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, N>
__esimd_wrregion(__SEIEED::vector_type_t<T, N> OldVal,
                 __SEIEED::vector_type_t<T, M> NewVal, uint16_t Offset,
                 __SEIEED::simd_mask_storage_t<M> Mask) {
  uint16_t EltOffset = Offset / sizeof(T);
  assert(Offset % sizeof(T) == 0);

  int NumRows = M / Width;
  assert(M % Width == 0);

  __SEIEED::vector_type_t<T, N> Result = OldVal;
  int Index = 0;
  for (int i = 0; i < NumRows; ++i) {
    for (int j = 0; j < Width; ++j) {
      if (Mask[Index])
        Result[i * VStride + j * Stride + EltOffset] = NewVal[Index];
      ++Index;
    }
  }
  return Result;
}

template <typename T, int N, int M, int ParentWidth>
__ESIMD_INTRIN __SEIEED::vector_type_t<T, N>
__esimd_wrindirect(__SEIEED::vector_type_t<T, N> OldVal,
                   __SEIEED::vector_type_t<T, M> NewVal,
                   __SEIEED::vector_type_t<uint16_t, M> Offset,
                   __SEIEED::simd_mask_storage_t<M> Mask) {
  __SEIEED::vector_type_t<T, N> Result = OldVal;
  for (int i = 0; i < M; ++i) {
    if (Mask[i]) {
      uint16_t EltOffset = Offset[i] / sizeof(T);
      assert(Offset[i] % sizeof(T) == 0);
      assert(EltOffset < N);
      Result[EltOffset] = NewVal[i];
    }
  }
  return Result;
}

#endif // __SYCL_DEVICE_ONLY__

/// @endcond ESIMD_DETAIL
