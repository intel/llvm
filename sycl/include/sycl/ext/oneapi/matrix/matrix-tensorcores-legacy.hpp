//===-------------- matrix-tensorcores-legacy.hpp - -----------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once
#include "sycl/detail/defines_elementary.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi {
namespace experimental::matrix {

enum class matrix_use { a, b, accumulator };

enum class matrix_layout { row_major, col_major, packed_a, packed_b };

namespace precision {
class tf32 {};
} // namespace precision

template <typename T, matrix_use Use, size_t Rows = sycl::dynamic_extent,
          size_t Cols = sycl::dynamic_extent,
          matrix_layout Layout = matrix_layout::row_major,
          typename Group = sycl::sub_group, typename Cond = void>
struct joint_matrix;

template <typename type, size_t size> class wi_data {
  marray<type, size> &data;
  wi_data(marray<type, size> &wi_data) : data(wi_data){};
  template <typename T, matrix_use Use, size_t Rows, size_t Cols,
            matrix_layout Layout, typename Group, typename Cond>
  friend struct joint_matrix;

public:
  size_t length() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return data.size();
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  };

  type &operator[](size_t i) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return data[i];
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  };
};

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR(type, use, M, N, size)                \
  template <matrix_layout Layout>                                              \
  struct joint_matrix<                                                         \
      type, matrix_use::use, M, N, Layout, sycl::sub_group,                    \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    marray<type, size> wi_marray;                                              \
    inline __SYCL_ALWAYS_INLINE wi_data<type, size> get_wi_data() {            \
      return wi_data(wi_marray);                                               \
    };                                                                         \
  };

// m8n32k16
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 8, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 8, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, accumulator, 8, 32, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, accumulator, 8, 32, 8)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 8, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 8, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 32, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int32_t, accumulator, 8, 32, 8)
// m32n8k16
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 8, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, accumulator, 32, 8, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, accumulator, 32, 8, 8)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 32, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int32_t, accumulator, 32, 8, 8)
// m16n16k16
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 16, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, accumulator, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(float, accumulator, 16, 16, 8)

__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 16, 8)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(int32_t, accumulator, 16, 16, 8)
// m8n8k4 double only
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, a, 8, 4, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, b, 4, 8, 1)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR(double, accumulator, 8, 8, 2)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(precision, use, M, N, type, \
                                                   size)                       \
  template <matrix_layout Layout>                                              \
  struct joint_matrix<                                                         \
      precision, matrix_use::use, M, N, Layout, sycl::sub_group,               \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    marray<type, size> wi_marray;                                              \
    inline __SYCL_ALWAYS_INLINE wi_data<type, size> get_wi_data() {            \
      return wi_data(wi_marray);                                               \
    };                                                                         \
  };
// m16n16k8 tf32 only
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(precision::tf32, a, 16, 8, float, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(precision::tf32, b, 8, 16, float, 4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION

#define __SYCL_JOINT_MATRIX_OVERLOAD(type, use, M, N, frag_type, frag_size)    \
  template <matrix_layout Layout>                                              \
  struct joint_matrix<                                                         \
      type, matrix_use::use, M, N, Layout, sycl::sub_group,                    \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    frag_type wi_marray[frag_size];                                            \
  };

// bf16 data format uint16_t implementation is deprecated
// m8n32k16
__SYCL_JOINT_MATRIX_OVERLOAD(uint16_t, a, 8, 16, int32_t, 2)
__SYCL_JOINT_MATRIX_OVERLOAD(uint16_t, b, 16, 32, int32_t, 8)
// m32n8k16
__SYCL_JOINT_MATRIX_OVERLOAD(uint16_t, a, 32, 16, int32_t, 8)
__SYCL_JOINT_MATRIX_OVERLOAD(uint16_t, b, 16, 8, int32_t, 2)
// m16n16k16
__SYCL_JOINT_MATRIX_OVERLOAD(uint16_t, a, 16, 16, int32_t, 4)
__SYCL_JOINT_MATRIX_OVERLOAD(uint16_t, b, 16, 16, int32_t, 4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD

template <typename Group, typename T, matrix_use Use, size_t NumRows,
          size_t NumCols, matrix_layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group sg,
                  joint_matrix<T, Use, NumRows, NumCols, Layout, Group> &res,
                  const T2 v) {
  // We kept the unused "sg" in joint_matrix_fill to match the other DPC++
  // functions
  std::ignore = sg;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  res.wi_marray = v;
#else
  std::ignore = res;
  std::ignore = v;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

} // namespace experimental::matrix

namespace detail {

template <typename S, typename T,
          sycl::ext::oneapi::experimental::matrix::matrix_use Use,
          size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space, access::decorated IsDecorated,
          typename Cond = void>
struct joint_matrix_load_impl {
  void load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                S, Use, NumRows, NumCols, Layout, sycl::sub_group> &res,
            multi_ptr<T, Space, IsDecorated> src, size_t stride);
};

template <sycl::ext::oneapi::experimental::matrix::matrix_layout Layout>
constexpr int get_layout_id();

template <>
constexpr int get_layout_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major>() {
  return 0;
}

template <>
constexpr int get_layout_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major>() {
  return 1;
}

template <typename S, typename T,
          sycl::ext::oneapi::experimental::matrix::matrix_use Use,
          size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space, access::decorated IsDecorated>
struct joint_matrix_load_impl<
    S, T, Use, NumRows, NumCols, Layout, Space, IsDecorated,
    typename std::enable_if_t<Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::row_major ||
                              Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::col_major>> {
  void load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                S, Use, NumRows, NumCols, Layout, sycl::sub_group> &res,
            multi_ptr<T, Space, IsDecorated> src, size_t stride) {
    if constexpr (std::is_same<std::remove_const_t<T>, uint16_t>::value ||
                  std::is_same<std::remove_const_t<T>,
                               sycl::ext::oneapi::bfloat16>::value) {
      auto tileptr = reinterpret_cast<const int32_t *>(src.get());
      auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
      if constexpr (NumRows == 16 && NumCols == 16) {
        if constexpr (Use ==
                      sycl::ext::oneapi::experimental::matrix::matrix_use::a) {
          __mma_bf16_m16n16k16_ld_a(destptr, tileptr, stride,
                                    get_layout_id<Layout>());
        } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                        matrix_use::b) {
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
    } else if constexpr (std::is_same<std::remove_const_t<T>, uint8_t>::value) {
      auto tileptr = reinterpret_cast<const int32_t *>(src.get());
      auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
      if constexpr (NumRows == 16 && NumCols == 16) {
        if constexpr (Use ==
                      sycl::ext::oneapi::experimental::matrix::matrix_use::a) {
          __imma_m16n16k16_ld_a_u8(destptr, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                        matrix_use::b) {
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
    } else if constexpr (std::is_same<std::remove_const_t<T>, int8_t>::value) {
      auto tileptr = reinterpret_cast<const int32_t *>(src.get());
      auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
      if constexpr (NumRows == 16 && NumCols == 16) {
        if constexpr (Use ==
                      sycl::ext::oneapi::experimental::matrix::matrix_use::a) {
          __imma_m16n16k16_ld_a_s8(destptr, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                        matrix_use::b) {
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
    } else if constexpr (std::is_same<std::remove_const_t<T>, half>::value) {
      auto tileptr = reinterpret_cast<const int32_t *>(src.get());
      auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
      if constexpr (NumRows == 16 && NumCols == 16) {
        if constexpr (Use ==
                      sycl::ext::oneapi::experimental::matrix::matrix_use::a) {
          __hmma_m16n16k16_ld_a(dstptr, tileptr, stride,
                                get_layout_id<Layout>());
        } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                        matrix_use::b) {
          __hmma_m16n16k16_ld_b(dstptr, tileptr, stride,
                                get_layout_id<Layout>());
        } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                        matrix_use::accumulator) {
          __hmma_m16n16k16_ld_c_f16(dstptr, tileptr, stride,
                                    get_layout_id<Layout>());
        }
      } else if constexpr (NumRows == 8 && NumCols == 16) {
        __hmma_m8n32k16_ld_a(dstptr, tileptr, stride, get_layout_id<Layout>());
      } else if constexpr (NumRows == 16 && NumCols == 32) {
        __hmma_m8n32k16_ld_b(dstptr, tileptr, stride, get_layout_id<Layout>());
      } else if constexpr (NumRows == 32 && NumCols == 16) {
        __hmma_m32n8k16_ld_a(dstptr, tileptr, stride, get_layout_id<Layout>());
      } else if constexpr (NumRows == 16 && NumCols == 8) {
        __hmma_m32n8k16_ld_b(dstptr, tileptr, stride, get_layout_id<Layout>());
      } else if constexpr (NumRows == 32 && NumCols == 8) {
        __hmma_m32n8k16_ld_c_f16(dstptr, tileptr, stride,
                                 get_layout_id<Layout>());
      } else if constexpr (NumRows == 8 && NumCols == 32) {
        __hmma_m8n32k16_ld_c_f16(dstptr, tileptr, stride,
                                 get_layout_id<Layout>());
      }

    } else if constexpr (std::is_same<std::remove_const_t<T>, int32_t>::value) {
      auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
      if constexpr (NumRows == 16 && NumCols == 16) {
        __imma_m16n16k16_ld_c(destptr, src.get(), stride,
                              get_layout_id<Layout>());
      } else if constexpr (NumRows == 8 && NumCols == 32) {
        __imma_m8n32k16_ld_c(destptr, src.get(), stride,
                             get_layout_id<Layout>());
      } else if constexpr (NumRows == 32 && NumCols == 8) {
        __imma_m32n8k16_ld_c(destptr, src.get(), stride,
                             get_layout_id<Layout>());
      }
    } else if constexpr (std::is_same<std::remove_const_t<T>, float>::value) {
      if constexpr (std::is_same<S, float>::value) {
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
      } else if constexpr (std::is_same<S,
                                        sycl::ext::oneapi::experimental::
                                            matrix::precision::tf32>::value) {
        auto tileptr = reinterpret_cast<const int32_t *>(src.get());
        auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
        if constexpr (NumRows == 16 && NumCols == 8) {
          __mma_tf32_m16n16k8_ld_a(dstptr, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (NumRows == 8 && NumCols == 16) {
          __mma_tf32_m16n16k8_ld_b(dstptr, tileptr, stride,
                                   get_layout_id<Layout>());
        }
      }
    } else if constexpr (std::is_same<std::remove_const_t<T>, double>::value) {
      auto dstptr = reinterpret_cast<double *>(&res.wi_marray);
      if constexpr (Use ==
                    sycl::ext::oneapi::experimental::matrix::matrix_use::a) {
        __dmma_m8n8k4_ld_a(dstptr, src.get(), stride, get_layout_id<Layout>());
      } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                      matrix_use::b) {
        __dmma_m8n8k4_ld_b(dstptr, src.get(), stride, get_layout_id<Layout>());
      } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                      matrix_use::accumulator) {
        __dmma_m8n8k4_ld_c(dstptr, src.get(), stride, get_layout_id<Layout>());
      }
    }
  }
};

template <typename T, size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space, access::decorated IsDecorated,
          typename Cond = void>
struct joint_matrix_store_impl {
  void
  store(sycl::ext::oneapi::experimental::matrix::joint_matrix<
            T, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
            NumRows, NumCols, Layout, sycl::sub_group> &src,
        multi_ptr<T, Space, IsDecorated> dst, size_t stride);
};

template <typename T, size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space, access::decorated IsDecorated>
struct joint_matrix_store_impl<
    T, NumRows, NumCols, Layout, Space, IsDecorated,
    typename std::enable_if_t<Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::row_major ||
                              Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::col_major>> {
  void
  store(sycl::ext::oneapi::experimental::matrix::joint_matrix<
            T, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
            NumRows, NumCols, Layout, sycl::sub_group> &src,
        multi_ptr<T, Space, IsDecorated> dst, size_t stride) {
    if constexpr (NumRows == 16 && NumCols == 16) {
      if constexpr (std::is_same<T, float>::value) {
        __hmma_m16n16k16_st_c_f32(dst.get(),
                                  reinterpret_cast<float *>(&src.wi_marray),
                                  stride, get_layout_id<Layout>());
      } else if constexpr (std::is_same<T, int32_t>::value) {
        __imma_m16n16k16_st_c_i32(dst.get(),
                                  reinterpret_cast<int32_t *>(&src.wi_marray),
                                  stride, get_layout_id<Layout>());
      } else if constexpr (std::is_same<T, half>::value) {
        __hmma_m16n16k16_st_c_f16(reinterpret_cast<int32_t *>(dst.get()),
                                  reinterpret_cast<int32_t *>(&src.wi_marray),
                                  stride, get_layout_id<Layout>());
      }
    } else if constexpr (NumRows == 8 && NumCols == 32) {
      if constexpr (std::is_same<T, float>::value) {
        __hmma_m8n32k16_st_c_f32(dst.get(),
                                 reinterpret_cast<float *>(&src.wi_marray),
                                 stride, get_layout_id<Layout>());
      } else if constexpr (std::is_same<T, int32_t>::value) {
        __imma_m8n32k16_st_c_i32(dst.get(),
                                 reinterpret_cast<int32_t *>(&src.wi_marray),
                                 stride, get_layout_id<Layout>());
      } else if constexpr (std::is_same<T, half>::value) {
        __hmma_m8n32k16_st_c_f16(reinterpret_cast<int32_t *>(dst.get()),
                                 reinterpret_cast<int32_t *>(&src.wi_marray),
                                 stride, get_layout_id<Layout>());
      }
    } else if constexpr (NumRows == 32 && NumCols == 8) {
      if constexpr (std::is_same<T, float>::value) {
        __hmma_m32n8k16_st_c_f32(dst.get(),
                                 reinterpret_cast<float *>(&src.wi_marray),
                                 stride, get_layout_id<Layout>());
      } else if constexpr (std::is_same<T, int32_t>::value) {
        __imma_m32n8k16_st_c_i32(dst.get(),
                                 reinterpret_cast<int32_t *>(&src.wi_marray),
                                 stride, get_layout_id<Layout>());
      } else if constexpr (std::is_same<T, half>::value) {
        __hmma_m32n8k16_st_c_f16(reinterpret_cast<int32_t *>(dst.get()),
                                 reinterpret_cast<int32_t *>(&src.wi_marray),
                                 stride, get_layout_id<Layout>());
      }
    } else if constexpr (std::is_same<T, double>::value) {
      __dmma_m8n8k4_st_c_f64(dst.get(),
                             reinterpret_cast<double *>(&src.wi_marray), stride,
                             get_layout_id<Layout>());
    }
  }
};

template <typename T1, typename T2, std::size_t M, std::size_t K, std::size_t N,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutB,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutC,
          typename Cond = void>
struct joint_matrix_mad_impl {
  sycl::ext::oneapi::experimental::matrix::joint_matrix<
      T2, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
      N, LayoutC, sycl::sub_group>
  mad(sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T1, sycl::ext::oneapi::experimental::matrix::matrix_use::a, M, K,
          LayoutA, sycl::sub_group>
          A,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T1, sycl::ext::oneapi::experimental::matrix::matrix_use::b, K, N,
          LayoutB, sycl::sub_group>
          B,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T2, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
          M, N, LayoutC, sycl::sub_group>
          C);
};

template <sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutB>
constexpr int get_layout_pair_id();

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major>() {
  return 0;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major>() {
  return 1;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major>() {
  return 2;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major>() {
  return 3;
}

template <typename T1, typename T2, std::size_t M, std::size_t K, std::size_t N,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutB,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutC>
struct joint_matrix_mad_impl<
    T1, T2, M, K, N, LayoutA, LayoutB, LayoutC,
    typename std::enable_if_t<
        (LayoutA == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        row_major ||
         LayoutA == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        col_major) &&
        (LayoutB == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        row_major ||
         LayoutB == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        col_major) &&
        (LayoutC == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        row_major ||
         LayoutC == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        col_major)>> {
  sycl::ext::oneapi::experimental::matrix::joint_matrix<
      T2, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
      N, LayoutC, sycl::sub_group>
  mad(sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T1, sycl::ext::oneapi::experimental::matrix::matrix_use::a, M, K,
          LayoutA, sycl::sub_group>
          A,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T1, sycl::ext::oneapi::experimental::matrix::matrix_use::b, K, N,
          LayoutB, sycl::sub_group>
          B,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T2, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
          M, N, LayoutC, sycl::sub_group>
          C) {
    sycl::ext::oneapi::experimental::matrix::joint_matrix<
        T2, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
        N, LayoutC, sycl::sub_group>
        D;
    if constexpr (M == 16 && N == 16 && K == 16) {
      if constexpr (std::is_same<T2, int32_t>::value) {
        auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
        auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
        auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
        auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
        if constexpr (std::is_same<T1, int8_t>::value) {
          __imma_m16n16k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                                  get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else if constexpr (std::is_same<T1, uint8_t>::value) {
          __imma_m16n16k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                                  get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same<T1, half>::value) {
        auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
        auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
        if constexpr (std::is_same<T2, float>::value) {
          __hmma_m16n16k16_mma_f32f32(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else if constexpr (std::is_same<T2, half>::value) {
          __hmma_m16n16k16_mma_f16f16(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same<T1, uint16_t>::value ||
                           std::is_same<T1,
                                        sycl::ext::oneapi::bfloat16>::value) {
        __mma_bf16_m16n16k16_mma_f32(
            reinterpret_cast<float *>(&D.wi_marray),
            reinterpret_cast<const int32_t *>(&A.wi_marray),
            reinterpret_cast<const int32_t *>(&B.wi_marray),
            reinterpret_cast<const float *>(&C.wi_marray),
            get_layout_pair_id<LayoutA, LayoutB>(), 0);
      }
    } else if constexpr (M == 8 && N == 32 && K == 16) {
      if constexpr (std::is_same<T2, int32_t>::value) {
        auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
        auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
        auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
        auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
        if constexpr (std::is_same<T1, int8_t>::value) {
          __imma_m8n32k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                                 get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else if constexpr (std::is_same<T1, uint8_t>::value) {
          __imma_m8n32k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                                 get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same<T1, half>::value) {
        auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
        auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
        if constexpr (std::is_same<T2, float>::value) {
          __hmma_m8n32k16_mma_f32f32(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else if constexpr (std::is_same<T2, half>::value) {
          __hmma_m8n32k16_mma_f16f16(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same<T1, uint16_t>::value ||
                           std::is_same<T1,
                                        sycl::ext::oneapi::bfloat16>::value) {
        __mma_bf16_m8n32k16_mma_f32(
            reinterpret_cast<float *>(&D.wi_marray),
            reinterpret_cast<const int32_t *>(&A.wi_marray),
            reinterpret_cast<const int32_t *>(&B.wi_marray),
            reinterpret_cast<const float *>(&C.wi_marray),
            get_layout_pair_id<LayoutA, LayoutB>(), 0);
      }
    } else if constexpr (M == 32 && N == 8 && K == 16) {
      if constexpr (std::is_same<T2, int32_t>::value) {
        auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
        auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
        auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
        auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
        if constexpr (std::is_same<T1, int8_t>::value) {
          __imma_m32n8k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                                 get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else if constexpr (std::is_same<T1, uint8_t>::value) {
          __imma_m32n8k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                                 get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (std::is_same<T1, uint16_t>::value ||
                           std::is_same<T1,
                                        sycl::ext::oneapi::bfloat16>::value) {
        __mma_bf16_m32n8k16_mma_f32(
            reinterpret_cast<float *>(&D.wi_marray),
            reinterpret_cast<const int32_t *>(&A.wi_marray),
            reinterpret_cast<const int32_t *>(&B.wi_marray),
            reinterpret_cast<const float *>(&C.wi_marray),
            get_layout_pair_id<LayoutA, LayoutB>(), 0);
      } else if constexpr (std::is_same<T1, half>::value) {
        auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
        auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
        if constexpr (std::is_same<T2, float>::value) {
          __hmma_m32n8k16_mma_f32f32(
              reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else if constexpr (std::is_same<T2, half>::value) {
          __hmma_m32n8k16_mma_f16f16(
              reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
              reinterpret_cast<const int32_t *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      }
    } else if constexpr (M == 16 && N == 16 && K == 8) {
      __mma_tf32_m16n16k8_mma_f32(reinterpret_cast<float *>(&D.wi_marray),
                                  reinterpret_cast<int32_t *>(&A.wi_marray),
                                  reinterpret_cast<int32_t *>(&B.wi_marray),
                                  reinterpret_cast<float *>(&C.wi_marray),
                                  get_layout_pair_id<LayoutA, LayoutB>(), 0);
    } else if constexpr (std::is_same<T1, double>::value) {
      __dmma_m8n8k4_mma_f64(reinterpret_cast<double *>(&D.wi_marray),
                            reinterpret_cast<const double *>(&A.wi_marray),
                            reinterpret_cast<const double *>(&B.wi_marray),
                            reinterpret_cast<const double *>(&C.wi_marray),
                            get_layout_pair_id<LayoutA, LayoutB>(), 0);
    }
    return D;
  }
};
} // namespace detail

namespace experimental::matrix {

template <
    typename Group, typename S, typename T, matrix_use Use, size_t NumRows,
    size_t NumCols, matrix_layout Layout, access::address_space Space,
    access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value ||
                         (std::is_same<S, precision::tf32>::value &&
                          std::is_same<std::remove_const_t<T>, float>::value),
                     bool> = true>
void joint_matrix_load(
    Group sg, joint_matrix<S, Use, NumRows, NumCols, Layout, Group> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sycl::ext::oneapi::detail::joint_matrix_load_impl<
      S, T, Use, NumRows, NumCols, Layout, Space, IsDecorated>{}
      .load(res, src, stride);
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = src;
  std::ignore = stride;
  throw runtime_error(
      "When using SYCL_EXT_ONEAPI_MATRIX_VERSION=3 joint_matrix_load is "
      "only supported by CUDA devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout, access::address_space Space,
          access::decorated IsDecorated>
void joint_matrix_store(Group sg,
                        joint_matrix<T, matrix_use::accumulator, NumRows,
                                     NumCols, Layout, Group> &src,
                        multi_ptr<T, Space, IsDecorated> dst, size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  sycl::ext::oneapi::detail::joint_matrix_store_impl<
      T, NumRows, NumCols, Layout, Space, IsDecorated>{}
      .store(src, dst, stride);
#else
  std::ignore = sg;
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw runtime_error(
      "When using SYCL_EXT_ONEAPI_MATRIX_VERSION=3 joint_matrix_store is "
      "only supported by CUDA devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

template <typename Group, typename T1, typename T2, std::size_t M,
          std::size_t K, std::size_t N, matrix_layout LayoutA,
          matrix_layout LayoutB, matrix_layout LayoutC>
joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group>
joint_matrix_mad(
    Group sg, joint_matrix<T1, matrix_use::a, M, K, LayoutA, Group> A,
    joint_matrix<T1, matrix_use::b, K, N, LayoutB, Group> B,
    joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group> C) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  return sycl::ext::oneapi::detail::joint_matrix_mad_impl<
             T1, T2, M, K, N, LayoutA, LayoutB, LayoutC>{}
      .mad(A, B, C);
#else
  std::ignore = sg;
  std::ignore = A;
  std::ignore = B;
  std::ignore = C;
  throw runtime_error(
      "When using SYCL_EXT_ONEAPI_MATRIX_VERSION=3 joint_matrix_mad is "
      "only supported by CUDA devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

// This function rounds the bottom 13 bits up or down, and then zeros out the
// bottom bits
inline __SYCL_ALWAYS_INLINE float round_to_tf32(float a) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  int32_t tmp_int = __nvvm_f2tf32_rna(a);
  return __nvvm_bitcast_i2f(tmp_int);
#else
  uint32_t tmp_uint = reinterpret_cast<uint32_t &>(a);
  tmp_uint += 0x1000u;
  tmp_uint &= 0xFFFFE000u;
  float ret = reinterpret_cast<float &>(tmp_uint);
  return ret;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

} // namespace experimental::matrix
} // namespace ext::oneapi
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
