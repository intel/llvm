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
template <> struct spv_scope_traits<sycl::sub_group> {
  constexpr static auto value = __spv::Scope::Subgroup;
};
template <int D> struct spv_scope_traits<sycl::group<D>> {
  constexpr static auto value = __spv::Scope::Workgroup;
};

template <typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          typename Group = sycl::sub_group>
class wi_slice;

template <typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          typename Group = sycl::sub_group>
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

  inline __SYCL_ALWAYS_INLINE wi_slice<T, NumRows, NumCols, Layout, Group>
  get_wi_data() {
    return wi_slice<T, NumRows, NumCols, Layout, Group>(*this);
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

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group sg,
                  joint_matrix<T, NumRows, NumCols, Layout, Group> &res,
                  const T v) {
  // We kept the unused "sg" in joint_matrix_fill to match the other DPC++
  // functions
  (void)sg;
#ifdef __SYCL_DEVICE_ONLY__
  res.spvm = __spirv_CompositeConstruct<T, NumRows, NumCols>(v);
#else
  (void)res;
  (void)v;
#endif // __SYCL_DEVICE_ONLY__
}

template <typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          typename Group = sycl::sub_group>
class wi_element {
  joint_matrix<T, NumRows, NumCols, Layout, Group> &M;
  std::size_t idx;

public:
  wi_element(joint_matrix<T, NumRows, NumCols, Layout, Group> &Mat,
             std::size_t i)
      : M(Mat), idx(i) {}
  operator T() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_VectorExtractDynamic(M.spvm, idx);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
  wi_element &operator=(const T &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(M.spvm, rhs, idx);
    return *this;
#else
    (void)rhs;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
  wi_element &operator*=(const T &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(
        M.spvm, __spirv_VectorExtractDynamic(M.spvm, idx) * rhs, idx);
    return *this;
#else
    (void)rhs;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
  // TODO: add other arithmetic operators
};

template <typename T, size_t NumRows, size_t NumCols, matrix_layout Layout,
          typename Group>
class wi_slice {
  joint_matrix<T, NumRows, NumCols, Layout, Group> &M;

public:
  wi_slice(joint_matrix<T, NumRows, NumCols, Layout, Group> &Mat) : M(Mat) {}
  size_t length() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_JointMatrixWorkItemLengthINTEL(M.spvm);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
  wi_element<T, NumRows, NumCols, Layout, Group> operator[](size_t i) {
    return wi_element<T, NumRows, NumCols, Layout, Group>(M, i);
  }
};

} // namespace experimental::matrix
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
