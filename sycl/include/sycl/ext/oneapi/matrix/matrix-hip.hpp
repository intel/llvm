
//===-------- matrix-hip.hpp - matrix ext impl ---*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once
#include "matrix-unified-utils.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>

#if defined(__gfx90a__)
#define __HIP_PLATFORM_AMD_MFMA__
#endif

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {} // namespace matrix
} // namespace experimental

using matrix_layout = sycl::ext::oneapi::experimental::matrix::layout;
using matrix_use = sycl::ext::oneapi::experimental::matrix::use;

namespace detail {

template <typename T, matrix_use Use, size_t Rows, size_t Cols,
          matrix_layout Layout = matrix_layout::dynamic, typename Cond = void>
struct joint_matrix_hip;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__HIP_PLATFORM_AMD_MFMA__)

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
  template <matrix_layout Layout>                                              \
  struct joint_matrix_hip<                                                     \
      TYPE, matrix_use::USE, M, N, Layout,                                     \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    using ext_array_t = __attribute__((                                        \
        __vector_size__(SIZE * sizeof(typename to_hip_type<TYPE>::type))))     \
    typename to_hip_type<TYPE>::type;                                          \
    ext_array_t data = {0};                                                    \
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

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(USE, M, N, SIZE)                 \
  template <matrix_layout Layout>                                              \
  struct joint_matrix_hip<                                                     \
      int8_t, matrix_use::USE, M, N, Layout,                                   \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    int8_t data[SIZE];                                                         \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(a, 32, 8, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(b, 8, 32, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(a, 16, 16, 4)
__SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR(b, 16, 16, 4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_INT8_ARR

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(TYPE, M, N)                       \
  template <>                                                                  \
  struct joint_matrix_hip<TYPE, matrix_use::accumulator, M, N,                 \
                          matrix_layout::dynamic> {                            \
    using ext_array_t =                                                        \
        __attribute__((__vector_size__((M * N) / 64 * sizeof(TYPE)))) TYPE;    \
    ext_array_t data = {0};                                                    \
  };

__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(float, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(double, 16, 16)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 32, 32)
__SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(int32_t, 16, 16)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC

template <matrix_layout Layout, typename S, typename T, size_t M, size_t N,
          access::address_space Space, access::decorated IsDecorated,
          typename Group>
void load_accumulator_layoutT(joint_matrix_hip<S, matrix_use::accumulator, M, N,
                                               matrix_layout::dynamic> &res,
                              multi_ptr<T, Space, IsDecorated> src,
                              size_t stride, Group &sg) {
  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();

  if constexpr (std::is_same_v<S, double>) {
    const auto thread_x = idx % N;
    const auto thread_y = idx / N;

    if constexpr (Layout == matrix_layout::row_major) {
      for (int i = 0; i < 4; ++i) {
        const int s_idx = thread_x + i * 4 * stride + thread_y * stride;
        res.data[i] = src[s_idx];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int s_idx = i * 4 + thread_x * stride + thread_y;
        res.data[i] = src[s_idx];
      }
    }
  } else if constexpr (std::is_same_v<S, float> || std::is_same_v<S, int32_t>) {
    if constexpr (M == 16 && N == 16) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == matrix_layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int s_idx = thread_x + i * stride + thread_y * 4 * stride;
          res.data[i] = src[s_idx];
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          const int s_idx = i + thread_x * stride + thread_y * 4;
          res.data[i] = src[s_idx];
        }
      }
    } else if constexpr (M == 32 && N == 32) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == matrix_layout::row_major) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_idx =
                thread_x + i * stride + thread_y * 4 * stride + j * 8 * N;
            res.data[i + 4 * j] = src[s_idx];
          }
        }
      } else {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int s_idx = i + thread_x * stride + thread_y * 4 + j * 8;
            res.data[i + 4 * j] = src[s_idx];
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
void load_accumulator_hip(joint_matrix_hip<S, matrix_use::accumulator, M, N,
                                           matrix_layout::dynamic> &res,
                          multi_ptr<T, Space, IsDecorated> src, size_t stride,
                          matrix_layout layout, Group &sg) {
  static_assert(std::is_same_v<S, int32_t> || std::is_same_v<S, float> ||
                    std::is_same_v<S, double>,
                "Unsupported matrix type!");

  if (layout == matrix_layout::row_major)
    load_accumulator_layoutT<matrix_layout::row_major>(res, src, stride, sg);
  else
    load_accumulator_layoutT<matrix_layout::col_major>(res, src, stride, sg);
}

template <typename Group, typename S, typename T, size_t M, size_t N,
          matrix_use Use, matrix_layout Layout, access::address_space Space,
          access::decorated IsDecorated,
          typename = typename std::enable_if_t<
              (Layout == matrix_layout::row_major ||
               Layout == matrix_layout::col_major) &&
              std::is_same_v<S, std::remove_const_t<T>>>>
void load_multiplicand_hip(joint_matrix_hip<S, Use, M, N, Layout> &res,
                           multi_ptr<T, Space, IsDecorated> src, size_t stride,
                           Group &sg) {
  static_assert(std::is_same_v<S, half> || std::is_same_v<S, bfloat16> ||
                    std::is_same_v<S, int8_t> || std::is_same_v<S, double>,
                "Unsupported matrix type!");

  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();

  if constexpr (std::is_same_v<S, double>) {
    if constexpr (Layout == matrix_layout::row_major) {
      res.data[0] = src[idx];
    } else if constexpr (Layout == matrix_layout::col_major) {
      res.data[0] = src[(idx % M) * 4 + idx / M];
    }
  } else {
    constexpr int Dim = (M == 16) ? 16 : 32;

    const auto thread_x = idx % Dim;
    const auto thread_y = idx / Dim;

    if constexpr (Layout == matrix_layout::col_major) {
      for (int i = 0; i < 4; ++i) {
        const int c_idx = thread_x * stride + i + thread_y * 4;
        res.data[i] = src[c_idx];
      }
    } else if constexpr (Layout == matrix_layout::row_major) {
      for (int i = 0; i < 4; ++i) {
        const int r_idx = thread_x + i * stride + thread_y * stride * 4;
        res.data[i] = src[r_idx];
      }
    }
  }
}

template <typename Group, matrix_layout Layout, typename T, size_t M, size_t N,
          access::address_space Space, access::decorated IsDecorated>
void store_layoutT(joint_matrix_hip<T, matrix_use::accumulator, M, N,
                                    matrix_layout::dynamic> &src,
                   multi_ptr<T, Space, IsDecorated> dst, size_t stride,
                   Group &sg) {
  const auto idx = sg.get_group_linear_id() * sg.get_local_range()[0] +
                   sg.get_local_linear_id();

  if constexpr (std::is_same_v<T, double>) {
    const auto thread_x = idx % N;
    const auto thread_y = idx / N;

    if constexpr (Layout == matrix_layout::row_major) {
      for (int i = 0; i < 4; ++i) {
        const int d_idx = thread_x + i * 4 * stride + thread_y * stride;
        dst[d_idx] = src.data[i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int d_idx = i * 4 + thread_x * stride + thread_y;
        dst[d_idx] = src.data[i];
      }
    }
  } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t>) {
    if constexpr (M == 16 && N == 16) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == matrix_layout::row_major) {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = thread_x + i * stride + thread_y * 4 * stride;
          dst[d_idx] = src.data[i];
        }
      } else {
        for (int i = 0; i < 4; ++i) {
          const int d_idx = i + thread_x * stride + thread_y * 4;
          dst[d_idx] = src.data[i];
        }
      }
    } else if constexpr (M == 32 && N == 32) {
      const auto thread_x = idx % N;
      const auto thread_y = idx / N;

      if constexpr (Layout == matrix_layout::row_major) {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int d_idx =
                thread_x + i * stride + thread_y * 4 * stride + j * 8 * stride;
            dst[d_idx] = src.data[i + 4 * j];
          }
        }
      } else {
        for (int j = 0; j < 4; ++j) {
          for (int i = 0; i < 4; ++i) {
            const int d_idx = i + thread_x * stride + thread_y * 4 + j * 8;
            dst[d_idx] = src.data[i + 4 * j];
          }
        }
      }
    }
  }
}

template <typename Group, typename T, size_t M, size_t N,
          access::address_space Space, access::decorated IsDecorated>
void joint_matrix_store_hip(joint_matrix_hip<T, matrix_use::accumulator, M, N,
                                             matrix_layout::dynamic> &src,
                            multi_ptr<T, Space, IsDecorated> dst, size_t stride,
                            matrix_layout layout, Group &sg) {
  if (matrix_layout::row_major == layout) {
    store_layoutT<Group, matrix_layout::row_major>(src, dst, stride, sg);
  } else {
    store_layoutT<Group, matrix_layout::col_major>(src, dst, stride, sg);
  }
}

template <typename Tm, typename Tc, std::size_t M, std::size_t K, std::size_t N,
          matrix_layout LayoutA, matrix_layout LayoutB,
          std::enable_if_t<(LayoutA == matrix_layout::row_major ||
                            LayoutA == matrix_layout::col_major) &&
                               (LayoutB == matrix_layout::row_major ||
                                LayoutB == matrix_layout::col_major),
                           bool> = true>
void joint_matrix_mad_hip(joint_matrix_hip<Tc, matrix_use::accumulator, M, N,
                                           matrix_layout::dynamic> &D,
                          joint_matrix_hip<Tm, matrix_use::a, M, K, LayoutA> &A,
                          joint_matrix_hip<Tm, matrix_use::b, K, N, LayoutB> &B,
                          joint_matrix_hip<Tc, matrix_use::accumulator, M, N,
                                           matrix_layout::dynamic> &C) {
  if constexpr (std::is_same_v<Tm, sycl::half>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_f32_16x16x16f16(A.data, B.data, C.data, 0,
                                                     0, 0);
    } else if constexpr (M == 32 && N == 32) {
      D.data =
          __builtin_amdgcn_mfma_f32_32x32x8f16(A.data, B.data, C.data, 0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, bfloat16>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(A.data, B.data, C.data,
                                                         0, 0, 0);
    } else if constexpr (M == 32 && N == 32) {
      D.data = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(A.data, B.data, C.data,
                                                        0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, double>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_f64_16x16x4f64(A.data[0], B.data[0],
                                                    C.data, 0, 0, 0);
    }
  } else if constexpr (std::is_same_v<Tm, int8_t>) {
    if constexpr (M == 16 && N == 16) {
      D.data = __builtin_amdgcn_mfma_i32_16x16x16i8(
          *reinterpret_cast<int32_t *>(A.data),
          *reinterpret_cast<int32_t *>(B.data), C.data, 0, 0, 0);
    } else if constexpr (M == 32 && N == 32) {
      D.data = __builtin_amdgcn_mfma_i32_32x32x8i8(
          *reinterpret_cast<int32_t *>(A.data),
          *reinterpret_cast<int32_t *>(B.data), C.data, 0, 0, 0);
    }
  } else {
    static_assert(false && "Invalid configuration!");
  }
}

template <typename S, size_t M, size_t N, matrix_use Use, matrix_layout Layout,
          typename F>
void joint_matrix_apply(joint_matrix_hip<S, Use, M, N, Layout> &jm,
                        F &&lambda) {
  if constexpr (std::is_same_v<S, double> && Use != matrix_use::accumulator) {
    jm.data[0] = lambda(jm.data[0]);
  } else if constexpr (Use != matrix_use::accumulator ||
                       (Use == matrix_use::accumulator && NumRows == 16)) {
    for (auto i = 0; i < 4; ++i)
      jm.data[i] = lambda(jm.data[i]);
  } else {
    for (auto i = 0; i < 16; ++i)
      jm.data[i] = lambda(jm.data[i]);
  }
}

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

} // namespace detail
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
