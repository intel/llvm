
//===-------- matrix-hip.hpp - matrix ext impl ---*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#include "matrix-unified-utils.hpp"

#include <sycl/access/access.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/marray.hpp>
#include <sycl/multi_ptr.hpp>

#include <cstring>

#define __HIP_PLATFORM_AMD_MFMA__

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace detail {

constexpr int WAVEFRONT_SIZE = 64;

template <typename T, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t Rows, size_t Cols,
          sycl::ext::oneapi::experimental::matrix::layout Layout =
              sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          typename Cond = void>
struct joint_matrix_hip;

using bfloat16x4 = __attribute__((__vector_size__(4 * sizeof(__bf16)))) __fp16;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int;
using int32x16 = __attribute__((__vector_size__(16 * sizeof(int32_t)))) int;
using doublex4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;
using half_x8 = _Float16 __attribute__((ext_vector_type(8)));
using bfloat16_x8 = __bf16 __attribute__((ext_vector_type(8)));
using int32_x4 = int __attribute__((ext_vector_type(4)));
using int32_x8 = int __attribute__((ext_vector_type(8)));

// OCP 8-bit and 4-bit floating-point element types. They are only consumed by
// the gfx950 (CDNA4) MFMA instructions, whose FP8/BF8 encodings follow the OCP
// E4M3/E5M2 formats used by sycl_ext_oneapi_fp8. The gfx94x (CDNA3) FP8 MFMAs
// use the incompatible E4M3FNUZ/E5M2FNUZ encodings and are not exposed.
using fp8_e4m3_t = sycl::ext::oneapi::experimental::fp8_e4m3_x<1>;
using fp8_e5m2_t = sycl::ext::oneapi::experimental::fp8_e5m2_x<1>;
using fp4_e2m1_t = sycl::ext::oneapi::experimental::fp4_e2m1_x<1>;

template <typename T>
inline constexpr bool is_hip_fp8_v =
    std::is_same_v<T, fp8_e4m3_t> || std::is_same_v<T, fp8_e5m2_t>;

template <typename T>
inline constexpr bool is_hip_small_float_v =
    is_hip_fp8_v<T> || std::is_same_v<T, fp4_e2m1_t>;

// The gfx950 FP8/FP4 MFMAs take independent A and B formats, so these are the
// only multiplicand types that may differ between A and B.
template <typename Ta, typename Tb>
inline constexpr bool is_hip_mixed_mfma_v =
    is_hip_small_float_v<Ta> && is_hip_small_float_v<Tb>;

// Format selector for the CBSZ (A) / BLGP (B) operands of the
// V_MFMA_F32_{16X16X128,32X32X64}_F8F6F4 instructions.
template <typename T> constexpr int f8f6f4_format() {
  if constexpr (std::is_same_v<T, fp8_e4m3_t>)
    return 0;
  else if constexpr (std::is_same_v<T, fp8_e5m2_t>)
    return 1;
  else
    return 4;
}

template <typename T> struct to_hip_type {
  using type = T;
};

template <> struct to_hip_type<bfloat16> {
  using type = __bf16;
};

template <> struct to_hip_type<half> {
  using type = __fp16;
};

template <> struct to_hip_type<int8_t> {
  using type = int32_t;
};

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR(TYPE, USE, M, N, SIZE)                \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_hip<                                                     \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::USE, M, N, Layout,   \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    sycl::marray<TYPE, SIZE> wi_marray{};                                      \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 8, 32, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 8, 32, 4)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, a, 16, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, b, 4, 16, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, a, 32, 2, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, b, 2, 32, 1)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 16, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 4, 16, 1)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 8, 32, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 16, 4)

// gfx942 (CDNA3) int8 MFMA uses larger K dimensions with packed operands:
// 16x16x32 and 32x32x16. Each work-item holds 8 int8 values (one i64).
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 32, 8)

// gfx950 (CDNA4) doubles K for the fp16/bf16 (16x16x32, 32x32x16) and int8
// (16x16x64, 32x32x32) MFMAs: 8 fp16/bf16 or 16 int8 values per work-item.
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 32, 8)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 32, 8)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 64, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 64, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 32, 32, 16)

// gfx950 FP8 (E4M3/E5M2) MFMAs: 16x16x32 and 32x32x16 (8 values per
// work-item), plus the F8F6F4 shapes 16x16x128 and 32x32x64 (32 values per
// work-item), which are shared with FP4 (E2M1).
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, a, 16, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, b, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, a, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, b, 16, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, a, 16, 128, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, b, 128, 16, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, a, 32, 64, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e4m3_t, b, 64, 32, 32)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, a, 16, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, b, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, a, 32, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, b, 16, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, a, 16, 128, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, b, 128, 16, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, a, 32, 64, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp8_e5m2_t, b, 64, 32, 32)

// FP4 values are held one per byte (as fp4_e2m1 stores them in memory) and are
// packed two per byte when the MFMA is issued.
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp4_e2m1_t, a, 16, 128, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp4_e2m1_t, b, 128, 16, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp4_e2m1_t, a, 32, 64, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(fp4_e2m1_t, b, 64, 32, 32)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N)                       \
  template <>                                                                  \
  struct joint_matrix_hip<                                                     \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,   \
      sycl::ext::oneapi::experimental::matrix::layout::dynamic> {              \
    sycl::marray<TYPE, (M * N) / WAVEFRONT_SIZE> wi_marray{};                  \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(double, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 16, 16)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC

template <sycl::ext::oneapi::experimental::matrix::layout Layout, typename S,
          typename T, size_t M, size_t N, access::address_space Space,
          access::decorated IsDecorated, typename Group>
void load_accumulator_layoutT(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, Group &sg) {
  // The fragment lane index is the work-item's position within its own
  // sub-group (0 .. sub_group_size-1). It must NOT include the sub-group's
  // index within the work-group, otherwise kernels that launch more than one
  // sub-group per work-group compute out-of-range element offsets.
  const auto idx = sg.get_local_linear_id();

  if constexpr (std::is_same_v<S, double>) {
    const auto thread_x = idx % N;
    const auto thread_y = idx / N;

    if constexpr (Layout ==
                  sycl::ext::oneapi::experimental::matrix::layout::row_major) {
      for (int i = 0; i < 4; ++i) {
        const int s_idx = thread_x + i * 4 * stride + thread_y * stride;
        res.wi_marray[i] = src[s_idx];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int s_idx = i * 4 + thread_x * stride + thread_y;
        res.wi_marray[i] = src[s_idx];
      }
    }
  } else if constexpr (std::is_same_v<S, float> || std::is_same_v<S, int32_t>) {
    if constexpr (M == 16 && N == 16) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int i = 0; i < 4; ++i) {
          const int s_idx = thread_x + i * stride + thread_y * 4 * stride;
          res.wi_marray[i] = src[s_idx];
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          const int s_idx = i + thread_x * stride + thread_y * 4;
          res.wi_marray[i] = src[s_idx];
        }
      }
    } else if constexpr (M == 32 && N == 32) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_idx =
                thread_x + i * stride + thread_y * 4 * stride + j * 8 * stride;
            res.wi_marray[i + 4 * j] = src[s_idx];
          }
        }
      } else {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_idx = i + thread_x * stride + thread_y * 4 + j * 8;
            res.wi_marray[i + 4 * j] = src[s_idx];
          }
        }
      }
    }
  }
}

template <
    typename Group, typename S, typename T, size_t M, size_t N,
    access::address_space Space, access::decorated IsDecorated,
    typename = std::enable_if_t<std::is_same_v<S, std::remove_const_t<T>>>>
void load_accumulator_hip(
    joint_matrix_hip<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout layout, Group &sg) {
  if (layout == sycl::ext::oneapi::experimental::matrix::layout::row_major)
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::row_major>(res, src,
                                                                    stride, sg);
  else
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::col_major>(res, src,
                                                                    stride, sg);
}

template <
    typename Group, typename S, typename T, size_t M, size_t N,
    sycl::ext::oneapi::experimental::matrix::use Use,
    sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    typename = typename std::enable_if_t<
        (Layout == sycl::ext::oneapi::experimental::matrix::layout::row_major ||
         Layout ==
             sycl::ext::oneapi::experimental::matrix::layout::col_major) &&
        std::is_same_v<S, std::remove_const_t<T>>>>
void load_multiplicand_hip(joint_matrix_hip<S, Use, M, N, Layout> &res,
                           multi_ptr<T, Space, IsDecorated> src, size_t stride,
                           Group &sg) {
  // The fragment lane index is the work-item's position within its own
  // sub-group (0 .. sub_group_size-1). It must NOT include the sub-group's
  // index within the work-group, otherwise kernels that launch more than one
  // sub-group per work-group compute out-of-range element offsets.
  const auto idx = sg.get_local_linear_id();

  // The AMD MFMA fragment layout is described in terms of the matrix's "free"
  // dimension (rows M for the A multiplicand, columns N for the B multiplicand)
  // and the shared contraction dimension K. Whether the free dimension is the
  // strided one in memory depends on BOTH the use and the layout:
  //   - A row_major: (m,k) at m*stride + k -> free dim (m) is strided.
  //   - A col_major: (m,k) at k*stride + m -> free dim (m) is contiguous.
  //   - B row_major: (k,n) at k*stride + n -> free dim (n) is contiguous.
  //   - B col_major: (k,n) at n*stride + k -> free dim (n) is strided.
  // So the strided-free-dim access pattern applies to (A, row_major) and
  // (B, col_major); the contiguous-free-dim pattern applies to (A, col_major)
  // and (B, row_major). Selecting the pattern from this combined condition
  // (rather than from the layout alone) makes both layouts behave correctly for
  // the A multiplicand, which previously loaded A transposed for row_major.
  constexpr bool StridedFreeDim =
      (Use == sycl::ext::oneapi::experimental::matrix::use::a)
          ? (Layout ==
             sycl::ext::oneapi::experimental::matrix::layout::row_major)
          : (Layout ==
             sycl::ext::oneapi::experimental::matrix::layout::col_major);

  // Free dimension: M for the A multiplicand, N for the B multiplicand. The
  // wavefront is split into (WAVEFRONT_SIZE / FreeDim) groups along the
  // contraction dimension K, and each work-item holds one contiguous K-block of
  // `Size` elements (Size == 1 for the double shapes).
  constexpr int FreeDim =
      (Use == sycl::ext::oneapi::experimental::matrix::use::a) ? M : N;
  constexpr int Size = (M * N) / WAVEFRONT_SIZE;

  const auto thread_x = idx % FreeDim;
  const auto thread_y = idx / FreeDim;

  // Position along K of the work-item's i-th element. FP8 operands of the
  // F8F6F4 MFMAs (32 elements per work-item) are split in two 16-element
  // halves, the second one offset by K / 2, rather than being one contiguous
  // K-block like FP4. Matching the hardware layout keeps K consistent when
  // FP8 is mixed with FP4.
  auto k_index = [&](int i) -> int {
    if constexpr (is_hip_fp8_v<S> && Size == 32) {
      constexpr int KGroups = WAVEFRONT_SIZE / FreeDim;
      return (i / 16) * (KGroups * 16) + thread_y * 16 + i % 16;
    } else {
      return thread_y * Size + i;
    }
  };

  if constexpr (StridedFreeDim) {
    for (int i = 0; i < Size; ++i) {
      const int c_idx = thread_x * stride + k_index(i);
      res.wi_marray[i] = src[c_idx];
    }
  } else {
    for (int i = 0; i < Size; ++i) {
      const int r_idx = thread_x + k_index(i) * stride;
      res.wi_marray[i] = src[r_idx];
    }
  }
}

template <typename Group,
          sycl::ext::oneapi::experimental::matrix::layout Layout, typename T,
          size_t M, size_t N, access::address_space Space,
          access::decorated IsDecorated>
void store_layoutT(
    const joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride, Group &sg) {
  // The fragment lane index is the work-item's position within its own
  // sub-group (0 .. sub_group_size-1). It must NOT include the sub-group's
  // index within the work-group, otherwise kernels that launch more than one
  // sub-group per work-group compute out-of-range element offsets.
  const auto idx = sg.get_local_linear_id();

  if constexpr (std::is_same_v<T, double>) {
    const auto thread_x = idx % N;
    const auto thread_y = idx / N;

    if constexpr (Layout ==
                  sycl::ext::oneapi::experimental::matrix::layout::row_major) {
      for (int i = 0; i < 4; ++i) {
        const int d_idx = thread_x + i * 4 * stride + thread_y * stride;
        dst[d_idx] = src.wi_marray[i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int d_idx = i * 4 + thread_x * stride + thread_y;
        dst[d_idx] = src.wi_marray[i];
      }
    }
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>) {
    if constexpr (M == 16 && N == 16) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = thread_x + i * stride + thread_y * 4 * stride;
          dst[d_idx] = src.wi_marray[i];
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = i + thread_x * stride + thread_y * 4;
          dst[d_idx] = src.wi_marray[i];
        }
      }
    } else if constexpr (M == 32 && N == 32) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == sycl::ext::oneapi::experimental::matrix::layout::
                                  row_major) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int d_idx =
                thread_x + i * stride + thread_y * 4 * stride + j * 8 * stride;
            dst[d_idx] = src.wi_marray[i + 4 * j];
          }
        }
      } else {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int d_idx = i + thread_x * stride + thread_y * 4 + j * 8;
            dst[d_idx] = src.wi_marray[i + 4 * j];
          }
        }
      }
    }
  }
}

template <typename Group, typename T, size_t M, size_t N,
          access::address_space Space, access::decorated IsDecorated>
void joint_matrix_store_hip(
    const joint_matrix_hip<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout layout, Group &sg) {
  if (sycl::ext::oneapi::experimental::matrix::layout::row_major == layout) {
    store_layoutT<Group,
                  sycl::ext::oneapi::experimental::matrix::layout::row_major>(
        src, dst, stride, sg);
  } else {
    store_layoutT<Group,
                  sycl::ext::oneapi::experimental::matrix::layout::col_major>(
        src, dst, stride, sg);
  }
}

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||       \
    defined(__gfx950__)
#define __SYCL_HIP_CDNA3_MFMA__
#endif

// Builds the 256-bit A/B operand of the gfx950 F8F6F4 MFMAs from the 32 values
// a work-item holds. FP8 values are already one per byte; FP4 values are packed
// two per byte with the lower K index in the low nibble, filling the low 128
// bits (the instruction ignores the upper registers for FP4).
template <typename T, size_t Size>
int32_x8 make_f8f6f4_operand(const sycl::marray<T, Size> &Vals) {
  static_assert(Size == 32, "F8F6F4 MFMA operands hold 32 values per lane");
  int32_x8 Operand{};
  if constexpr (std::is_same_v<T, fp4_e2m1_t>) {
    uint8_t Packed[Size / 2];
    for (size_t i = 0; i < Size / 2; ++i)
      Packed[i] = static_cast<uint8_t>((Vals[2 * i].vals[0] & 0xF) |
                                       (Vals[2 * i + 1].vals[0] << 4));
    std::memcpy(&Operand, Packed, sizeof(Packed));
  } else {
    std::memcpy(&Operand, &Vals, Size);
  }
  return Operand;
}

template <typename Ta, typename Tb, typename Tc, std::size_t M, std::size_t K,
          std::size_t N,
          sycl::ext::oneapi::experimental::matrix::layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::layout LayoutB>
void joint_matrix_mad_hip(
    joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &D,
    const joint_matrix_hip<Ta, sycl::ext::oneapi::experimental::matrix::use::a,
                           M, K, LayoutA> &A,
    const joint_matrix_hip<Tb, sycl::ext::oneapi::experimental::matrix::use::b,
                           K, N, LayoutB> &B,
    const joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &C) {
#if defined(__gfx950__)
  constexpr bool HasSmallFloatMFMA = true;
#else
  constexpr bool HasSmallFloatMFMA = false;
#endif
  static_assert(!is_hip_mixed_mfma_v<Ta, Tb> || HasSmallFloatMFMA,
                "joint_matrix with fp8_e4m3, fp8_e5m2 or fp4_e2m1 "
                "multiplicands is only supported on AMD gfx950");
#if defined(__gfx90a__) || defined(__SYCL_HIP_CDNA3_MFMA__)
  if constexpr (std::is_same_v<Ta, float>) {
#if defined(__SYCL_HIP_CDNA3_MFMA__)
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x4f32(
          A.wi_marray[0], B.wi_marray[0],
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x2f32(
          A.wi_marray[0], B.wi_marray[0],
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
#endif
  } else if constexpr (std::is_same_v<Ta, sycl::half>) {
    if constexpr (M == 16 && N == 16 && K == 16) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x16f16(
          *reinterpret_cast<const float16x4 *>(&A.wi_marray),
          *reinterpret_cast<const float16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32 && K == 8) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x8f16(
          *reinterpret_cast<const float16x4 *>(&A.wi_marray),
          *reinterpret_cast<const float16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
#if defined(__gfx950__)
    else if constexpr (M == 16 && N == 16 && K == 32) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x32_f16(
          *reinterpret_cast<const half_x8 *>(&A.wi_marray),
          *reinterpret_cast<const half_x8 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32 && K == 16) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x16_f16(
          *reinterpret_cast<const half_x8 *>(&A.wi_marray),
          *reinterpret_cast<const half_x8 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
#endif
  } else if constexpr (std::is_same_v<Ta, bfloat16>) {
    if constexpr (M == 16 && N == 16 && K == 16) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          *reinterpret_cast<const bfloat16x4 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32 && K == 8) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
          *reinterpret_cast<const bfloat16x4 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
#if defined(__gfx950__)
    else if constexpr (M == 16 && N == 16 && K == 32) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
          *reinterpret_cast<const bfloat16_x8 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16_x8 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32 && K == 16) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
          *reinterpret_cast<const bfloat16_x8 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16_x8 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
#endif
  } else if constexpr (std::is_same_v<Ta, double>) {
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_f64_16x16x4f64(
          A.wi_marray[0], B.wi_marray[0],
          *reinterpret_cast<const doublex4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(double));
    }
  } else if constexpr (std::is_same_v<Ta, int8_t>) {
#if defined(__SYCL_HIP_CDNA3_MFMA__)
    // CDNA3 (gfx942) int8 MFMA: 16x16x32 / 32x32x16 with packed i64 operands
    // (8 int8 values per work-item).
    if constexpr (M == 16 && N == 16 && K == 32) {
      auto result = __builtin_amdgcn_mfma_i32_16x16x32_i8(
          *reinterpret_cast<const int64_t *>(&A.wi_marray),
          *reinterpret_cast<const int64_t *>(&B.wi_marray),
          *reinterpret_cast<const int32x4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(int32_t));
    } else if constexpr (M == 32 && N == 32 && K == 16) {
      auto result = __builtin_amdgcn_mfma_i32_32x32x16_i8(
          *reinterpret_cast<const int64_t *>(&A.wi_marray),
          *reinterpret_cast<const int64_t *>(&B.wi_marray),
          *reinterpret_cast<const int32x16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(int32_t));
    }
#if defined(__gfx950__)
    // CDNA4 (gfx950) int8 MFMA: 16x16x64 / 32x32x32 with 128-bit operands
    // (16 int8 values per work-item).
    else if constexpr (M == 16 && N == 16 && K == 64) {
      auto result = __builtin_amdgcn_mfma_i32_16x16x64_i8(
          *reinterpret_cast<const int32_x4 *>(&A.wi_marray),
          *reinterpret_cast<const int32_x4 *>(&B.wi_marray),
          *reinterpret_cast<const int32x4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(int32_t));
    } else if constexpr (M == 32 && N == 32 && K == 32) {
      auto result = __builtin_amdgcn_mfma_i32_32x32x32_i8(
          *reinterpret_cast<const int32_x4 *>(&A.wi_marray),
          *reinterpret_cast<const int32_x4 *>(&B.wi_marray),
          *reinterpret_cast<const int32x16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(int32_t));
    }
#endif
#else
    // CDNA2 (gfx90a) int8 MFMA: 16x16x16 / 32x32x8 with i32 operands
    // (4 int8 values per work-item).
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_i32_16x16x16i8(
          *reinterpret_cast<const Tc *>(&A.wi_marray),
          *reinterpret_cast<const Tc *>(&B.wi_marray),
          *reinterpret_cast<const int32x4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(int32_t));
    } else if constexpr (M == 32 && N == 32) {
      auto result = __builtin_amdgcn_mfma_i32_32x32x8i8(
          *reinterpret_cast<const Tc *>(&A.wi_marray),
          *reinterpret_cast<const Tc *>(&B.wi_marray),
          *reinterpret_cast<const int32x16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(int32_t));
    }
#endif
  }
#if defined(__gfx950__)
  else if constexpr (is_hip_mixed_mfma_v<Ta, Tb>) {
    if constexpr ((M == 16 && K == 32) || (M == 32 && K == 16)) {
      // FP8 MFMAs with 64-bit operands (8 values per work-item). On gfx950
      // these opcodes interpret FP8/BF8 as OCP E4M3/E5M2.
      const auto a = *reinterpret_cast<const int64_t *>(&A.wi_marray);
      const auto b = *reinterpret_cast<const int64_t *>(&B.wi_marray);
      constexpr bool AIsE4M3 = std::is_same_v<Ta, fp8_e4m3_t>;
      constexpr bool BIsE4M3 = std::is_same_v<Tb, fp8_e4m3_t>;
      if constexpr (M == 16) {
        const auto c = *reinterpret_cast<const floatx4 *>(&C.wi_marray);
        floatx4 result;
        if constexpr (AIsE4M3 && BIsE4M3)
          result = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0);
        else if constexpr (AIsE4M3)
          result = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a, b, c, 0, 0, 0);
        else if constexpr (BIsE4M3)
          result = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a, b, c, 0, 0, 0);
        else
          result = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a, b, c, 0, 0, 0);
        std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
      } else {
        const auto c = *reinterpret_cast<const floatx16 *>(&C.wi_marray);
        floatx16 result;
        if constexpr (AIsE4M3 && BIsE4M3)
          result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c, 0, 0, 0);
        else if constexpr (AIsE4M3)
          result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a, b, c, 0, 0, 0);
        else if constexpr (BIsE4M3)
          result = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a, b, c, 0, 0, 0);
        else
          result = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a, b, c, 0, 0, 0);
        std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
      }
    } else if constexpr ((M == 16 && K == 128) || (M == 32 && K == 64)) {
      // F8F6F4 MFMAs. CBSZ/BLGP select the A/B formats; the scale operands are
      // zero, which selects the unscaled V_MFMA_F32_*_F8F6F4 encoding.
      const int32_x8 a = make_f8f6f4_operand(A.wi_marray);
      const int32_x8 b = make_f8f6f4_operand(B.wi_marray);
      constexpr int FmtA = f8f6f4_format<Ta>();
      constexpr int FmtB = f8f6f4_format<Tb>();
      if constexpr (M == 16) {
        auto result = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a, b, *reinterpret_cast<const floatx4 *>(&C.wi_marray), FmtA, FmtB,
            0, 0, 0, 0);
        std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
      } else {
        auto result = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a, b, *reinterpret_cast<const floatx16 *>(&C.wi_marray), FmtA, FmtB,
            0, 0, 0, 0);
        std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
      }
    }
  }
#endif // __gfx950__
#endif // __gfx90a__ || __SYCL_HIP_CDNA3_MFMA__
}

#undef __SYCL_HIP_CDNA3_MFMA__

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
