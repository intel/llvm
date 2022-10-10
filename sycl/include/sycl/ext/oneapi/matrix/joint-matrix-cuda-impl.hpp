// joint-matrix-cuda-impl.hpp - joint_matrix cuda specializations-*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-----------------------------------------------------------------------=== //

#include <sycl/ext/oneapi/matrix/joint-matrix.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
  namespace ext {
  namespace oneapi {
  namespace experimental {
  namespace matrix {

  template <typename type, size_t size> class wi_data {
    marray<type, size> &data;
    wi_data(marray<type, size> &wi_data) : data(wi_data){};
    template <typename T, matrix_use Use, size_t Rows, size_t Cols,
              layout Layout, typename Group, typename Cond>
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
  template <layout Layout>                                                     \
  struct joint_matrix<                                                         \
      type, matrix_use::use, M, N, Layout, sycl::sub_group,                    \
      typename std::enable_if_t<Layout == layout::row_major ||                 \
                                Layout == layout::col_major>> {                \
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

  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 8, 16, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 32, 16)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 8, 16, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 32, 16)
  // m32n8k16
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 32, 16, 16)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 8, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, a, 32, 16, 16)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(half, b, 16, 8, 16)

  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, a, 32, 16, 16)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(int8_t, b, 16, 8, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, a, 32, 16, 16)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(uint8_t, b, 16, 8, 4)
  // m16n16k16
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, a, 16, 16, 8)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR(bfloat16, b, 16, 16, 8)
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

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_ACC(type, M, N, size)                 \
  template <>                                                                  \
  struct joint_matrix<type, matrix_use::accumulator, M, N, layout::dynamic,    \
                      sycl::sub_group> {                                       \
    marray<type, size> wi_marray;                                              \
    inline __SYCL_ALWAYS_INLINE wi_data<type, size> get_wi_data() {            \
      return wi_data(wi_marray);                                               \
    };                                                                         \
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

#define __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(precision, use, M, N, type, \
                                                   size)                       \
  template <layout Layout>                                                     \
  struct joint_matrix<                                                         \
      precision, matrix_use::use, M, N, Layout, sycl::sub_group,               \
      typename std::enable_if_t<Layout == layout::row_major ||                 \
                                Layout == layout::col_major>> {                \
    marray<type, size> wi_marray;                                              \
    inline __SYCL_ALWAYS_INLINE wi_data<type, size> get_wi_data() {            \
      return wi_data(wi_marray);                                               \
    };                                                                         \
  };
  // m16n16k8 tf32 only
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(precision::tf32, a, 16, 8, float,
                                             4)
  __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION(precision::tf32, b, 8, 16, float,
                                             4)

#undef __SYCL_JOINT_MATRIX_OVERLOAD_ARR_PRECISION

  } // namespace matrix
  } // namespace experimental
  } // namespace oneapi
  } // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
