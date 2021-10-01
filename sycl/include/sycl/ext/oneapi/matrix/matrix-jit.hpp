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
#include <CL/sycl/detail/defines_sycl_ns.hpp>
#include <CL/sycl/feature_test.hpp>

__SYCL_OPEN_NS() {
namespace ext {
namespace oneapi {
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

template <typename G> struct spv_scope_traits {};
template <> struct spv_scope_traits<__sycl_ns::sub_group> {
  constexpr static auto value = __spv::Scope::Subgroup;
};
template <int D> struct spv_scope_traits<__sycl_ns::group<D>> {
  constexpr static auto value = __spv::Scope::Workgroup;
};

template <typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          typename Group = __sycl_ns::sub_group>
struct joint_matrix {
public:
  __spv::__spirv_JointMatrixINTEL<
      T, NumRows, NumCols, spv_matrix_layout_traits<Layout>::value> *spvm;
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
inline __SYCL_ALWAYS_INLINE void
joint_matrix_load(Group sg,
                  joint_matrix<T, NumRows, NumCols, Layout, Group> &res,
                  multi_ptr<T, Space> src, size_t stride, matrix_layout MemL) {
#ifdef __SYCL_DEVICE_ONLY__
  T *Ptr = src.get();
  switch (MemL) {
  default:
    assert(false && "Invalid Memory Layout!");
  case matrix_layout::row_major:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
            Ptr, stride, __spv::MatrixLayout::RowMajor,
            spv_scope_traits<Group>::value);
    break;
  case matrix_layout::col_major:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
            Ptr, stride, __spv::MatrixLayout::ColumnMajor,
            spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_a:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
            Ptr, stride, __spv::MatrixLayout::PackedA,
            spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_b:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
            Ptr, stride, __spv::MatrixLayout::PackedB,
            spv_scope_traits<Group>::value);
    break;
  }
#else
  (void)sg;
  (void)res;
  (void)src;
  (void)stride;
  (void)MemL;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout MatL = matrix_layout::row_major,
          access::address_space Space>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_store(Group sg,
                   joint_matrix<T, NumRows, NumCols, MatL, Group> &src,
                   multi_ptr<T, Space> res, size_t stride, matrix_layout MemL) {
#ifdef __SYCL_DEVICE_ONLY__
  T *Ptr = res.get();
  switch (MemL) {
  default:
    assert(false && "Invalid Memory Layout!");
  case matrix_layout::row_major:
    __spirv_JointMatrixStoreINTEL<T, NumRows, NumCols,
                                  spv_matrix_layout_traits<MatL>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case matrix_layout::col_major:
    __spirv_JointMatrixStoreINTEL<T, NumRows, NumCols,
                                  spv_matrix_layout_traits<MatL>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_a:
    __spirv_JointMatrixStoreINTEL<T, NumRows, NumCols,
                                  spv_matrix_layout_traits<MatL>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::PackedA,
        spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_b:
    __spirv_JointMatrixStoreINTEL<T, NumRows, NumCols,
                                  spv_matrix_layout_traits<MatL>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::PackedB,
        spv_scope_traits<Group>::value);
    break;
  }
#else
  (void)sg;
  (void)src;
  (void)res;
  (void)stride;
  (void)MemL;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename Group, typename T1, typename T2, typename T3, size_t M,
          size_t K, size_t N, matrix_layout LayoutA, matrix_layout LayoutB,
          matrix_layout LayoutC>
inline __SYCL_ALWAYS_INLINE joint_matrix<T3, M, N, LayoutC, Group>
joint_matrix_mad(Group sg, joint_matrix<T1, M, K, LayoutA, Group> &mA,
                 joint_matrix<T2, K, N, LayoutB, Group> &mB,
                 joint_matrix<T3, M, N, LayoutC, Group> &mC) {
#ifdef __SYCL_DEVICE_ONLY__
  joint_matrix<T3, M, N, LayoutC, Group> res(sg);
  if constexpr (std::is_same<T1, uint16_t>::value &&
                std::is_same<T2, uint16_t>::value &&
                std::is_same<T3, float>::value)
    res.spvm = __spirv_JointMatrixMadINTEL(mA.spvm, mB.spvm, mC.spvm);
  else if constexpr (std::is_unsigned<T1>::value && std::is_unsigned<T2>::value)
    res.spvm = __spirv_JointMatrixUUMadINTEL(mA.spvm, mB.spvm, mC.spvm);
  else if constexpr (std::is_signed<T1>::value && std::is_unsigned<T2>::value)
    res.spvm = __spirv_JointMatrixSUMadINTEL(mA.spvm, mB.spvm, mC.spvm);
  else if constexpr (std::is_unsigned<T1>::value && std::is_signed<T2>::value)
    res.spvm = __spirv_JointMatrixUSMadINTEL(mA.spvm, mB.spvm, mC.spvm);
  else
    res.spvm = __spirv_JointMatrixMadINTEL(mA.spvm, mB.spvm, mC.spvm);
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
} // namespace oneapi
} // namespace ext
} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()
