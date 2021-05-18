//==------------------ matrix.hpp - SYCL matrix ----------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/detail/defines_elementary.hpp>
#include <CL/sycl/feature_test.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental::matrix {

enum class matrix_layout { row_major, col_major, packed_a, packed_b };

template <matrix_layout Layout> struct spv_matrix_layout_traits {
  static constexpr __spv::MatrixLayout value = __spv::MatrixLayout::RowMajor;
};

#define SPV_MATRIX_LAYOUT_TRAITS(LAYOUT, SPV_LAYOUT)                           \
  template <> struct spv_matrix_layout_traits<LAYOUT> {                        \
    static constexpr __spv::MatrixLayout value = SPV_LAYOUT;                   \
  };

SPV_MATRIX_LAYOUT_TRAITS(matrix_layout::row_major,
                         __spv::MatrixLayout::RowMajor)
SPV_MATRIX_LAYOUT_TRAITS(matrix_layout::col_major,
                         __spv::MatrixLayout::ColumnMajor)
SPV_MATRIX_LAYOUT_TRAITS(matrix_layout::packed_a, __spv::MatrixLayout::PackedA)
SPV_MATRIX_LAYOUT_TRAITS(matrix_layout::packed_b, __spv::MatrixLayout::PackedB)

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major>
struct joint_matrix {
public:
  __spv::__spirv_MatrixINTEL<T, NumRows, NumCols,
                             spv_matrix_layout_traits<Layout>::value> *spvm;
  joint_matrix(Group sg) {
#ifndef __SYCL_DEVICE_ONLY__
    (void)sg;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
};

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          access::address_space Space>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg, joint_matrix<Group, T, NumRows, NumCols, Layout> &res,
    multi_ptr<T, Space> src, size_t stride, matrix_layout L = Layout) {
#ifdef __SYCL_DEVICE_ONLY__
  T *Ptr = src.get();
  res.spvm = __spirv_MatrixLoadINTEL<T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
      Ptr, stride, spv_matrix_layout_traits<Layout>::value);
#else
  (void)sg;
  (void)res;
  (void)src;
  (void)stride;
  (void)L;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          access::address_space Space>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group sg, joint_matrix<Group, T, NumRows, NumCols, Layout> &res,
    multi_ptr<T, Space> src, size_t stride, matrix_layout L = Layout) {
#ifdef __SYCL_DEVICE_ONLY__
  T *Ptr = src.get();
  __spirv_MatrixStoreINTEL<T, NumRows, NumCols,
                           spv_matrix_layout_traits<Layout>::value>(
      Ptr, res.spvm, stride, spv_matrix_layout_traits<Layout>::value);
#else
  (void)sg;
  (void)res;
  (void)src;
  (void)stride;
  (void)L;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename Group, typename T1, typename T2, size_t M, size_t K,
          size_t N, matrix_layout LayoutA, matrix_layout LayoutB,
          matrix_layout LayoutC>
inline __SYCL_ALWAYS_INLINE joint_matrix<Group, T2, M, N, LayoutC>
joint_matrix_mad(Group sg, joint_matrix<Group, T1, M, K, LayoutA> &mA,
                 joint_matrix<Group, T1, K, N, LayoutB> &mB,
                 joint_matrix<Group, T2, M, N, LayoutC> &mC) {
#ifdef __SYCL_DEVICE_ONLY__
  joint_matrix<Group, T2, M, N, LayoutC> res(sg);
  res.spvm = __spirv_MatrixMadINTEL(mA.spvm, mB.spvm, mC.spvm);
  return res;
#else
  (void)sg;
  (void)mA;
  (void)mB;
  (void)mC;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}
} // namespace experimental::matrix
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
