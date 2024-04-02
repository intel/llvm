
//===-------- matrix-tensorcores.hpp - matrix ext impl ---*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#include "matrix-unified-utils.hpp"

#include <sycl/aliases.hpp>             // for half
#include <sycl/ext/oneapi/bfloat16.hpp> // for bfloat16
#include <sycl/half_type.hpp>           // for half
#include <sycl/marray.hpp>              // for marray

#include <stddef.h>    // for size_t
#include <stdint.h>    // for int8_t, uint8_t, int32_t
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout = layout::dynamic>
struct joint_matrix;

} // namespace matrix
} // namespace experimental

namespace detail {

template <typename T, sycl::ext::oneapi::experimental::matrix::use Use,
          size_t Rows, size_t Cols,
          sycl::ext::oneapi::experimental::matrix::layout Layout =
              sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          typename Cond = void>
struct joint_matrix_cuda;

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR(TYPE, USE, M, N, SIZE)                \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_cuda<                                                    \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::USE, M, N, Layout,   \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    marray<TYPE, SIZE> wi_marray;                                              \
  };

// m8n32k16
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(sycl::ext::oneapi::bfloat16, a, 8, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(sycl::ext::oneapi::bfloat16, b, 16, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 8, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 32, 16)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 8, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 8, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 32, 16)
// m32n8k16
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(sycl::ext::oneapi::bfloat16, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(sycl::ext::oneapi::bfloat16, b, 16, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 8, 16)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 8, 4)
// m16n16k16
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(sycl::ext::oneapi::bfloat16, a, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(sycl::ext::oneapi::bfloat16, b, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 16, 16)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 16, 8)
// m8n8k4 double only
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 8, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 4, 8, 1)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N, SIZE)                 \
  template <>                                                                  \
  struct joint_matrix_cuda<                                                    \
      TYPE, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,   \
      sycl::ext::oneapi::experimental::matrix::layout::dynamic> {              \
    marray<TYPE, SIZE> wi_marray;                                              \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(half, 8, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 8, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 8, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(half, 32, 8, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 8, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 32, 8, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(half, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(double, 8, 8, 2)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(PRECISION, USE, M, N, TYPE, \
                                                   SIZE)                       \
  template <sycl::ext::oneapi::experimental::matrix::layout Layout>            \
  struct joint_matrix_cuda<                                                    \
      PRECISION, sycl::ext::oneapi::experimental::matrix::use::USE, M, N,      \
      Layout,                                                                  \
      typename std::enable_if_t<                                               \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||    \
          Layout ==                                                            \
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {   \
    marray<TYPE, SIZE> wi_marray;                                              \
  };
// m16n16k8 tf32 only
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(
    sycl::ext::oneapi::experimental::matrix::precision::tf32, a, 16, 8, float,
    4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(
    sycl::ext::oneapi::experimental::matrix::precision::tf32, b, 8, 16, float,
    4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
template <sycl::ext::oneapi::experimental::matrix::layout Layout>
constexpr int get_layout_id();

template <>
constexpr int
get_layout_id<sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
  return 0;
}

template <>
constexpr int
get_layout_id<sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
  return 1;
}

template <sycl::ext::oneapi::experimental::matrix::layout Layout, typename S,
          typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
void load_accumulator_layoutT(
    joint_matrix_cuda<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride) {
  if constexpr (std::is_same_v<S, int32_t>) {
    auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
    if constexpr (NumRows == 16 && NumCols == 16) {
      __imma_m16n16k16_ld_c(destptr, src.get(), stride,
                            get_layout_id<Layout>());
    } else if constexpr (NumRows == 8 && NumCols == 32) {
      __imma_m8n32k16_ld_c(destptr, src.get(), stride, get_layout_id<Layout>());
    } else if constexpr (NumRows == 32 && NumCols == 8) {
      __imma_m32n8k16_ld_c(destptr, src.get(), stride, get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<S, float>) {
    auto dstptr = reinterpret_cast<float *>(&res.wi_marray);
    if constexpr (NumRows == 16 && NumCols == 16) {
      __hmma_m16n16k16_ld_c_f32(dstptr, src.get(), stride,
                                get_layout_id<Layout>());
    } else if constexpr (NumRows == 8 && NumCols == 32) {
      __hmma_m8n32k16_ld_c_f32(dstptr, src.get(), stride,
                               get_layout_id<Layout>());
    } else if constexpr (NumRows == 32 && NumCols == 8) {
      __hmma_m32n8k16_ld_c_f32(dstptr, src.get(), stride,
                               get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<S, half>) {
    auto tileptr = reinterpret_cast<const int32_t *>(src.get());
    auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
    if constexpr (NumRows == 32 && NumCols == 8) {
      __hmma_m32n8k16_ld_c_f16(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
    } else if constexpr (NumRows == 8 && NumCols == 32) {
      __hmma_m8n32k16_ld_c_f16(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 16) {
      __hmma_m16n16k16_ld_c_f16(dstptr, tileptr, stride,
                                get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<S, double>) {
    __dmma_m8n8k4_ld_c(reinterpret_cast<double *>(&res.wi_marray), src.get(),
                       stride, get_layout_id<Layout>());
  }
};

template <typename S, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
void load_accumulator_cuda(
    joint_matrix_cuda<
        S, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
  switch (Layout) {
  case sycl::ext::oneapi::experimental::matrix::layout::row_major:
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::row_major>(res, src,
                                                                    stride);
    break;
  case sycl::ext::oneapi::experimental::matrix::layout::col_major:
    load_accumulator_layoutT<
        sycl::ext::oneapi::experimental::matrix::layout::col_major>(res, src,
                                                                    stride);
    break;
  default:
    assert(false && "Invalid layout specified!");
  }
}

template <
    typename S, typename T, size_t NumRows, size_t NumCols,
    sycl::ext::oneapi::experimental::matrix::use Use,
    sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<
        Layout == sycl::ext::oneapi::experimental::matrix::layout::row_major ||
            Layout ==
                sycl::ext::oneapi::experimental::matrix::layout::col_major,
        bool> = true>
void load_multiplicand_cuda(
    joint_matrix_cuda<S, Use, NumRows, NumCols, Layout> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride) {
  if constexpr (std::is_same_v<S, sycl::ext::oneapi::bfloat16>) {
    auto tileptr = reinterpret_cast<const int32_t *>(src.get());
    auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
    if constexpr (NumRows == 16 && NumCols == 16) {
      if constexpr (Use == sycl::ext::oneapi::experimental::matrix::use::a) {
        __mma_bf16_m16n16k16_ld_a(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
      } else if constexpr (Use ==
                           sycl::ext::oneapi::experimental::matrix::use::b) {
        __mma_bf16_m16n16k16_ld_b(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
      }
    } else if constexpr (NumRows == 8 && NumCols == 16) {
      __mma_bf16_m8n32k16_ld_a(destptr, tileptr, stride,
                               get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 32) {
      __mma_bf16_m8n32k16_ld_b(destptr, tileptr, stride,
                               get_layout_id<Layout>());
    } else if constexpr (NumRows == 32 && NumCols == 16) {
      __mma_bf16_m32n8k16_ld_a(destptr, tileptr, stride,
                               get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 8) {
      __mma_bf16_m32n8k16_ld_b(destptr, tileptr, stride,
                               get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<S, uint8_t>) {
    auto tileptr = reinterpret_cast<const int32_t *>(src.get());
    auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
    if constexpr (NumRows == 16 && NumCols == 16) {
      if constexpr (Use == sycl::ext::oneapi::experimental::matrix::use::a) {
        __imma_m16n16k16_ld_a_u8(destptr, tileptr, stride,
                                 get_layout_id<Layout>());
      } else if constexpr (Use ==
                           sycl::ext::oneapi::experimental::matrix::use::b) {
        __imma_m16n16k16_ld_b_u8(destptr, tileptr, stride,
                                 get_layout_id<Layout>());
      }
    } else if constexpr (NumRows == 8 && NumCols == 16) {
      __imma_m8n32k16_ld_a_u8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 32) {
      __imma_m8n32k16_ld_b_u8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    } else if constexpr (NumRows == 32 && NumCols == 16) {
      __imma_m32n8k16_ld_a_u8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 8) {
      __imma_m32n8k16_ld_b_u8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<S, int8_t>) {
    auto tileptr = reinterpret_cast<const int32_t *>(src.get());
    auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
    if constexpr (NumRows == 16 && NumCols == 16) {
      if constexpr (Use == sycl::ext::oneapi::experimental::matrix::use::a) {
        __imma_m16n16k16_ld_a_s8(destptr, tileptr, stride,
                                 get_layout_id<Layout>());
      } else if constexpr (Use ==
                           sycl::ext::oneapi::experimental::matrix::use::b) {
        __imma_m16n16k16_ld_b_s8(destptr, tileptr, stride,
                                 get_layout_id<Layout>());
      }
    } else if constexpr (NumRows == 8 && NumCols == 16) {
      __imma_m8n32k16_ld_a_s8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 32) {
      __imma_m8n32k16_ld_b_s8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    } else if constexpr (NumRows == 32 && NumCols == 16) {
      __imma_m32n8k16_ld_a_s8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 8) {
      __imma_m32n8k16_ld_b_s8(destptr, tileptr, stride,
                              get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<S, half>) {
    auto tileptr = reinterpret_cast<const int32_t *>(src.get());
    auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
    if constexpr (NumRows == 16 && NumCols == 16) {
      if constexpr (Use == sycl::ext::oneapi::experimental::matrix::use::a) {
        __hmma_m16n16k16_ld_a(dstptr, tileptr, stride, get_layout_id<Layout>());
      } else if constexpr (Use ==
                           sycl::ext::oneapi::experimental::matrix::use::b) {
        __hmma_m16n16k16_ld_b(dstptr, tileptr, stride, get_layout_id<Layout>());
      }
    } else if constexpr (NumRows == 8 && NumCols == 16) {
      __hmma_m8n32k16_ld_a(dstptr, tileptr, stride, get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 32) {
      __hmma_m8n32k16_ld_b(dstptr, tileptr, stride, get_layout_id<Layout>());
    } else if constexpr (NumRows == 32 && NumCols == 16) {
      __hmma_m32n8k16_ld_a(dstptr, tileptr, stride, get_layout_id<Layout>());
    } else if constexpr (NumRows == 16 && NumCols == 8) {
      __hmma_m32n8k16_ld_b(dstptr, tileptr, stride, get_layout_id<Layout>());
    }

  } else if constexpr (std::is_same_v<S, sycl::ext::oneapi::experimental::
                                             matrix::precision::tf32>) {
    auto tileptr = reinterpret_cast<const int32_t *>(src.get());
    auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
    if constexpr (NumRows == 16 && NumCols == 8) {
      __mma_tf32_m16n16k8_ld_a(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
    } else if constexpr (NumRows == 8 && NumCols == 16) {
      __mma_tf32_m16n16k8_ld_b(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<S, double>) {
    auto dstptr = reinterpret_cast<double *>(&res.wi_marray);
    if constexpr (Use == sycl::ext::oneapi::experimental::matrix::use::a) {
      __dmma_m8n8k4_ld_a(dstptr, src.get(), stride, get_layout_id<Layout>());
    } else if constexpr (Use ==
                         sycl::ext::oneapi::experimental::matrix::use::b) {
      __dmma_m8n8k4_ld_b(dstptr, src.get(), stride, get_layout_id<Layout>());
    }
  }
}

template <sycl::ext::oneapi::experimental::matrix::layout Layout, typename T,
          size_t NumRows, size_t NumCols, access::address_space Space,
          access::decorated IsDecorated>
void store_layoutT(
    const joint_matrix_cuda<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride) {
  if constexpr (NumRows == 16 && NumCols == 16) {
    if constexpr (std::is_same_v<T, float>) {
      __hmma_m16n16k16_st_c_f32(dst.get(), &src.wi_marray[0], stride,
                                get_layout_id<Layout>());
    } else if constexpr (std::is_same_v<T, int32_t>) {
      __imma_m16n16k16_st_c_i32(dst.get(), &src.wi_marray[0], stride,
                                get_layout_id<Layout>());
    } else if constexpr (std::is_same_v<T, half>) {
      __hmma_m16n16k16_st_c_f16(
          reinterpret_cast<int32_t *>(dst.get()),
          reinterpret_cast<const int32_t *>(&src.wi_marray[0]), stride,
          get_layout_id<Layout>());
    }
  } else if constexpr (NumRows == 8 && NumCols == 32) {
    if constexpr (std::is_same_v<T, float>) {
      __hmma_m8n32k16_st_c_f32(dst.get(), &src.wi_marray[0], stride,
                               get_layout_id<Layout>());
    } else if constexpr (std::is_same_v<T, int32_t>) {
      __imma_m8n32k16_st_c_i32(dst.get(), &src.wi_marray[0], stride,
                               get_layout_id<Layout>());
    } else if constexpr (std::is_same_v<T, half>) {
      __hmma_m8n32k16_st_c_f16(
          reinterpret_cast<int32_t *>(dst.get()),
          reinterpret_cast<const int32_t *>(&src.wi_marray[0]), stride,
          get_layout_id<Layout>());
    }
  } else if constexpr (NumRows == 32 && NumCols == 8) {
    if constexpr (std::is_same_v<T, float>) {
      __hmma_m32n8k16_st_c_f32(dst.get(), &src.wi_marray[0], stride,
                               get_layout_id<Layout>());
    } else if constexpr (std::is_same_v<T, int32_t>) {
      __imma_m32n8k16_st_c_i32(dst.get(), &src.wi_marray[0], stride,
                               get_layout_id<Layout>());
    } else if constexpr (std::is_same_v<T, half>) {
      __hmma_m32n8k16_st_c_f16(
          reinterpret_cast<int32_t *>(dst.get()),
          reinterpret_cast<const int32_t *>(&src.wi_marray[0]), stride,
          get_layout_id<Layout>());
    }
  } else if constexpr (std::is_same_v<T, double>) {
    __dmma_m8n8k4_st_c_f64(dst.get(), &src.wi_marray[0], stride,
                           get_layout_id<Layout>());
  }
}

template <typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
void joint_matrix_store_cuda(
    const joint_matrix_cuda<
        T, sycl::ext::oneapi::experimental::matrix::use::accumulator, NumRows,
        NumCols, sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
  switch (Layout) {
  case sycl::ext::oneapi::experimental::matrix::layout::row_major:
    store_layoutT<sycl::ext::oneapi::experimental::matrix::layout::row_major>(
        src, dst, stride);
    break;
  case sycl::ext::oneapi::experimental::matrix::layout::col_major:
    store_layoutT<sycl::ext::oneapi::experimental::matrix::layout::col_major>(
        src, dst, stride);
    break;
  default:
    assert(false && "Invalid layout specified!");
  }
}

template <sycl::ext::oneapi::experimental::matrix::layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::layout LayoutB>
constexpr int get_layout_pair_id();

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::row_major,
    sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
  return 0;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::row_major,
    sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
  return 1;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::col_major,
    sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
  return 2;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::layout::col_major,
    sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
  return 3;
}

template <
    typename Tm, typename Tc, typename Td, std::size_t M, std::size_t K,
    std::size_t N, sycl::ext::oneapi::experimental::matrix::layout LayoutA,
    sycl::ext::oneapi::experimental::matrix::layout LayoutB,
    std::enable_if_t<
        (LayoutA ==
             sycl::ext::oneapi::experimental::matrix::layout::row_major ||
         LayoutA ==
             sycl::ext::oneapi::experimental::matrix::layout::col_major) &&
            (LayoutB ==
                 sycl::ext::oneapi::experimental::matrix::layout::row_major ||
             LayoutB ==
                 sycl::ext::oneapi::experimental::matrix::layout::col_major),
        bool> = true>
void joint_matrix_mad_cuda(
    joint_matrix_cuda<
        Td, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &D,
    const joint_matrix_cuda<Tm, sycl::ext::oneapi::experimental::matrix::use::a,
                            M, K, LayoutA> &A,
    const joint_matrix_cuda<Tm, sycl::ext::oneapi::experimental::matrix::use::b,
                            K, N, LayoutB> &B,
    const joint_matrix_cuda<
        Tc, sycl::ext::oneapi::experimental::matrix::use::accumulator, M, N,
        sycl::ext::oneapi::experimental::matrix::layout::dynamic> &C) {
  if constexpr (M == 16 && N == 16 && K == 16) {
    if constexpr (std::is_same_v<Tc, int32_t>) {
      auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
      auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
      auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
      auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
      if constexpr (std::is_same_v<Tm, int8_t>) {
        __imma_m16n16k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                                get_layout_pair_id<LayoutA, LayoutB>(), 0);
      } else if constexpr (std::is_same_v<Tm, uint8_t>) {
        __imma_m16n16k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                                get_layout_pair_id<LayoutA, LayoutB>(), 0);
      }
    } else if constexpr (std::is_same_v<Tm, half>) {
      auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
      auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
      if constexpr (std::is_same_v<Tc, float>) {
        if constexpr (std::is_same<Td, float>::value) {
          __hmma_m16n16k16_mma_f32f32(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else {
          __hmma_m16n16k16_mma_f16f32(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same_v<Tc, half>) {
        if constexpr (std::is_same<Td, float>::value) {
          __hmma_m16n16k16_mma_f32f16(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else {
          __hmma_m16n16k16_mma_f16f16(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      }
    } else if constexpr (std::is_same_v<Tm, sycl::ext::oneapi::bfloat16>) {
      __mma_bf16_m16n16k16_mma_f32(
          reinterpret_cast<float *>(&D.wi_marray),
          reinterpret_cast<const int32_t *>(&A.wi_marray),
          reinterpret_cast<const int32_t *>(&B.wi_marray),
          reinterpret_cast<const float *>(&C.wi_marray),
          get_layout_pair_id<LayoutA, LayoutB>(), 0);
    }
  } else if constexpr (M == 8 && N == 32 && K == 16) {
    if constexpr (std::is_same_v<Tc, int32_t>) {
      auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
      auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
      auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
      auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
      if constexpr (std::is_same_v<Tm, int8_t>) {
        __imma_m8n32k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                               get_layout_pair_id<LayoutA, LayoutB>(), 0);
      } else if constexpr (std::is_same_v<Tm, uint8_t>) {
        __imma_m8n32k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                               get_layout_pair_id<LayoutA, LayoutB>(), 0);
      }
    } else if constexpr (std::is_same_v<Tm, half>) {
      auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
      auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
      if constexpr (std::is_same_v<Tc, float>) {
        if constexpr (std::is_same<Td, float>::value) {
          __hmma_m8n32k16_mma_f32f32(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else {
          __hmma_m8n32k16_mma_f16f32(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same_v<Tc, half>) {
        if constexpr (std::is_same<Td, float>::value) {
          __hmma_m8n32k16_mma_f32f16(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else {
          __hmma_m8n32k16_mma_f16f16(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      }
    } else if constexpr (std::is_same_v<Tm, sycl::ext::oneapi::bfloat16>) {
      __mma_bf16_m8n32k16_mma_f32(
          reinterpret_cast<float *>(&D.wi_marray),
          reinterpret_cast<const int32_t *>(&A.wi_marray),
          reinterpret_cast<const int32_t *>(&B.wi_marray),
          reinterpret_cast<const float *>(&C.wi_marray),
          get_layout_pair_id<LayoutA, LayoutB>(), 0);
    }
  } else if constexpr (M == 32 && N == 8 && K == 16) {
    if constexpr (std::is_same_v<Tc, int32_t>) {
      auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
      auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
      auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
      auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
      if constexpr (std::is_same_v<Tm, int8_t>) {
        __imma_m32n8k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                               get_layout_pair_id<LayoutA, LayoutB>(), 0);
      } else if constexpr (std::is_same_v<Tm, uint8_t>) {
        __imma_m32n8k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                               get_layout_pair_id<LayoutA, LayoutB>(), 0);
      }
    } else if constexpr (std::is_same_v<Tm, sycl::ext::oneapi::bfloat16>) {
      __mma_bf16_m32n8k16_mma_f32(
          reinterpret_cast<float *>(&D.wi_marray),
          reinterpret_cast<const int32_t *>(&A.wi_marray),
          reinterpret_cast<const int32_t *>(&B.wi_marray),
          reinterpret_cast<const float *>(&C.wi_marray),
          get_layout_pair_id<LayoutA, LayoutB>(), 0);
    } else if constexpr (std::is_same_v<Tm, half>) {

      auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
      auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
      if constexpr (std::is_same_v<Tc, float>) {
        if constexpr (std::is_same<Td, float>::value) {
          __hmma_m32n8k16_mma_f32f32(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else {
          __hmma_m32n8k16_mma_f16f32(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same_v<Tc, half>) {
        if constexpr (std::is_same<Td, float>::value) {
          __hmma_m32n8k16_mma_f32f16(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else {
          __hmma_m32n8k16_mma_f16f16(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      }
    }
  } else if constexpr (M == 16 && N == 16 && K == 8) {
    __mma_tf32_m16n16k8_mma_f32(reinterpret_cast<float *>(&D.wi_marray),
                                reinterpret_cast<const int32_t *>(&A.wi_marray),
                                reinterpret_cast<const int32_t *>(&B.wi_marray),
                                reinterpret_cast<const float *>(&C.wi_marray),
                                get_layout_pair_id<LayoutA, LayoutB>(), 0);
  } else if constexpr (std::is_same_v<Tm, double>) {
    __dmma_m8n8k4_mma_f64(reinterpret_cast<double *>(&D.wi_marray),
                          reinterpret_cast<const double *>(&A.wi_marray),
                          reinterpret_cast<const double *>(&B.wi_marray),
                          reinterpret_cast<const double *>(&C.wi_marray),
                          get_layout_pair_id<LayoutA, LayoutB>(), 0);
  }
}

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
