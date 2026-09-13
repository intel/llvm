
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

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 16, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 4, 16, 1)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 8, 32, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 16, 4)

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
                thread_x + i * stride + thread_y * 4 * stride + j * 8 * N;
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
  //   - A row_major: element (m,k) at m*stride + k -> free dim (m) is strided.
  //   - A col_major: element (m,k) at k*stride + m -> free dim (m) is contiguous.
  //   - B row_major: element (k,n) at k*stride + n -> free dim (n) is contiguous.
  //   - B col_major: element (k,n) at n*stride + k -> free dim (n) is strided.
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

  if constexpr (StridedFreeDim) {
    for (int i = 0; i < Size; ++i) {
      const int c_idx = thread_x * stride + i + thread_y * Size;
      res.wi_marray[i] = src[c_idx];
    }
  } else {
    for (int i = 0; i < Size; ++i) {
      const int r_idx = thread_x + i * stride + thread_y * stride * Size;
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

template <typename Tm, typename Tc, std::size_t M, std::size_t K, std::size_t N,
          sycl::ext::oneapi::experimental::matrix::layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::layout LayoutB>
void joint_matrix_mad_hip(
    joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &D,
    const joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::a,
                           M, K, LayoutA> &A,
    const joint_matrix_hip<Tm, sycl::ext::oneapi::experimental::matrix::use::b,
                           K, N, LayoutB> &B,
    const joint_matrix_hip<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &C) {
#ifdef __gfx90a__
  if constexpr (std::is_same_v<Tm, sycl::half>) {
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x16f16(
          *reinterpret_cast<const float16x4 *>(&A.wi_marray),
          *reinterpret_cast<const float16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x8f16(
          *reinterpret_cast<const float16x4 *>(&A.wi_marray),
          *reinterpret_cast<const float16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
  } else if constexpr (std::is_same_v<Tm, bfloat16>) {
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
          *reinterpret_cast<const bfloat16x4 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(float));
    } else if constexpr (M == 32 && N == 32) {
      auto result = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
          *reinterpret_cast<const bfloat16x4 *>(&A.wi_marray),
          *reinterpret_cast<const bfloat16x4 *>(&B.wi_marray),
          *reinterpret_cast<const floatx16 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 16 * sizeof(float));
    }
  } else if constexpr (std::is_same_v<Tm, double>) {
    if constexpr (M == 16 && N == 16) {
      auto result = __builtin_amdgcn_mfma_f64_16x16x4f64(
          A.wi_marray[0], B.wi_marray[0],
          *reinterpret_cast<const doublex4 *>(&C.wi_marray), 0, 0, 0);
      std::memcpy(&D.wi_marray, &result, 4 * sizeof(double));
    }
  } else if constexpr (std::is_same_v<Tm, int8_t>) {
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
  }
#endif // __gfx90a__
}

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
