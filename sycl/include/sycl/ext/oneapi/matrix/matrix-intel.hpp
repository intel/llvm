//==------------------ matrix-jit-use.hpp - SYCL matrix ----------------*- C++
//-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/feature_test.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

// class tf32 should not hold actual data. It is a tag type only, an empty class
// with no member variables. Morally, it is equivalent to an enumeration--it
// just uses the type system to communicate the desired accuracy of arithmetic
// computations. Users can't construct a tf32
namespace precision {
class tf32 {
  tf32() = delete;
};
} // namespace precision

enum class layout { row_major, col_major, packed, dynamic };

template <layout Layout> struct spv_matrix_layout_traits {
  static constexpr __spv::MatrixLayout value = __spv::MatrixLayout::Dynamic;
};

#define SPV_MATRIX_LAYOUT_TRAITS(LAYOUT, SPV_LAYOUT)                           \
  template <> struct spv_matrix_layout_traits<LAYOUT> {                        \
    static constexpr __spv::MatrixLayout value = SPV_LAYOUT;                   \
  };

SPV_MATRIX_LAYOUT_TRAITS(layout::row_major, __spv::MatrixLayout::RowMajor)
SPV_MATRIX_LAYOUT_TRAITS(layout::col_major, __spv::MatrixLayout::ColumnMajor)
SPV_MATRIX_LAYOUT_TRAITS(layout::packed, __spv::MatrixLayout::Packed)
SPV_MATRIX_LAYOUT_TRAITS(layout::dynamic, __spv::MatrixLayout::Dynamic)

// unnecessary was introduced for backward compatibility.
// Once the use implementation is stable, "unnecessary" value will be omitted
enum class use { a, b, accumulator, unnecessary };

template <use Use> struct spv_matrix_use_traits {
  static constexpr __spv::MatrixUse value = __spv::MatrixUse::MatrixA;
};

#define SPV_MATRIX_USE_TRAITS(USE, SPV_USE)                                    \
  template <> struct spv_matrix_use_traits<USE> {                              \
    static constexpr __spv::MatrixUse value = SPV_USE;                         \
  };

SPV_MATRIX_USE_TRAITS(use::a, __spv::MatrixUse::MatrixA)
SPV_MATRIX_USE_TRAITS(use::b, __spv::MatrixUse::MatrixB)
SPV_MATRIX_USE_TRAITS(use::accumulator, __spv::MatrixUse::Accumulator)

template <typename G> struct spv_scope_traits {};
template <> struct spv_scope_traits<sycl::sub_group> {
  constexpr static auto value = __spv::Scope::Subgroup;
};
template <int D> struct spv_scope_traits<sycl::group<D>> {
  constexpr static auto value = __spv::Scope::Workgroup;
};

// forward declarations
template <typename T, use Use, size_t Rows, size_t Cols, layout Layout,
          typename Group>
struct joint_matrix;

template <typename T, size_t NumRows, size_t NumCols, use Use,
          layout Layout = layout::dynamic, typename Group = sycl::sub_group>
class wi_element {
  joint_matrix<T, Use, NumRows, NumCols, Layout, Group> &M;
  std::size_t idx;

public:
  wi_element(joint_matrix<T, Use, NumRows, NumCols, Layout, Group> &Mat,
             std::size_t i)
      : M(Mat), idx(i) {}
  operator T() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_VectorExtractDynamic(M.spvm, idx);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  explicit operator bool() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_VectorExtractDynamic(M.spvm, idx) != static_cast<T>(0);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  template <typename T2> wi_element &operator=(const T2 &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(M.spvm, static_cast<T>(rhs), idx);
    return *this;
#else
    (void)rhs;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &
  operator=(const wi_element<T, NumRows, NumCols, Use, Layout, Group> &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(
        M.spvm, __spirv_VectorExtractDynamic(rhs.M.spvm, rhs.idx), idx);
    return *this;
#else
    (void)rhs;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

#if __SYCL_DEVICE_ONLY__
#define OP(op)                                                                 \
  template <typename T2> wi_element &operator op##=(const T2 &rhs) {           \
    M.spvm = __spirv_VectorInsertDynamic(                                      \
        M.spvm,                                                                \
        static_cast<T>(__spirv_VectorExtractDynamic(M.spvm, idx)               \
                           op static_cast<T>(rhs)),                            \
        idx);                                                                  \
    return *this;                                                              \
  }
#else // __SYCL_DEVICE_ONLY__
#define OP(op)                                                                 \
  template <typename T2> wi_element &operator op##=(const T2 &rhs) {           \
    (void)rhs;                                                                 \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }
#endif // __SYCL_DEVICE_ONLY__
  OP(+)
  OP(-)
  OP(*)
  OP(/)
#undef OP
};

template <size_t NumRows, size_t NumCols, use Use, layout Layout,
          typename Group>
class wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows, NumCols,
                 Use, Layout, Group> {
  joint_matrix<sycl::ext::oneapi::experimental::bfloat16, Use, NumRows, NumCols,
               Layout, Group> &M;
  std::size_t idx;

public:
  wi_element(joint_matrix<sycl::ext::oneapi::experimental::bfloat16, Use,
                          NumRows, NumCols, Layout, Group> &Mat,
             std::size_t i)
      : M(Mat), idx(i) {}
  operator sycl::ext::oneapi::experimental::bfloat16() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_VectorExtractDynamic(M.spvm, idx);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  explicit operator bool() {
#ifdef __SYCL_DEVICE_ONLY__
    return std::fabs(static_cast<float>(__spirv_VectorExtractDynamic(
               M.spvm, idx))) >= std::numeric_limits<float>::epsilon();
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &operator=(const sycl::ext::oneapi::experimental::bfloat16 &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(M.spvm, rhs, idx);
    return *this;
#else
    (void)rhs;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &
  operator=(const wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows,
                             NumCols, Use, Layout, Group> &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(
        M.spvm, __spirv_VectorExtractDynamic(rhs.M.spvm, rhs.idx), idx);
    return *this;
#else
    (void)rhs;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

#if __SYCL_DEVICE_ONLY__
#define OP(opassign, op)                                                       \
  wi_element &operator opassign(                                               \
      const sycl::ext::oneapi::experimental::bfloat16 &rhs) {                  \
    M.spvm = __spirv_VectorInsertDynamic(                                      \
        M.spvm, __spirv_VectorExtractDynamic(M.spvm, idx) op rhs, idx);        \
    return *this;                                                              \
  }
#else // __SYCL_DEVICE_ONLY__
#define OP(opassign, op)                                                       \
  wi_element &operator opassign(                                               \
      const sycl::ext::oneapi::experimental::bfloat16 &rhs) {                  \
    (void)rhs;                                                                 \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }
#endif // __SYCL_DEVICE_ONLY__
  OP(+=, +)
  OP(-=, -)
  OP(*=, *)
  OP(/=, /)
#undef OP

#if __SYCL_DEVICE_ONLY__
#define OP(type, op)                                                           \
  friend type operator op(                                                     \
      const wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows,     \
                       NumCols, Use, Layout, Group> &lhs,                      \
      const sycl::ext::oneapi::experimental::bfloat16 &rhs) {                  \
    return __spirv_VectorExtractDynamic(lhs.M.spvm, lhs.idx) op rhs;           \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::experimental::bfloat16 &lhs,                    \
      const wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows,     \
                       NumCols, Use, Layout, Group> &rhs) {                    \
    return __spirv_VectorExtractDynamic(rhs.M.spvm, rhs.idx) op lhs;           \
  }
  OP(sycl::ext::oneapi::experimental::bfloat16, +)
  OP(sycl::ext::oneapi::experimental::bfloat16, -)
  OP(sycl::ext::oneapi::experimental::bfloat16, *)
  OP(sycl::ext::oneapi::experimental::bfloat16, /)
#undef OP
#define OP(type, op)                                                           \
  friend type operator op(                                                     \
      const wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows,     \
                       NumCols, Use, Layout, Group> &lhs,                      \
      const sycl::ext::oneapi::experimental::bfloat16 &rhs) {                  \
    return type{static_cast<float>(__spirv_VectorExtractDynamic(               \
        lhs.M.spvm, lhs.idx)) op static_cast<float>(rhs)};                     \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::experimental::bfloat16 &lhs,                    \
      const wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows,     \
                       NumCols, Use, Layout, Group> &rhs) {                    \
    return type{static_cast<float>(__spirv_VectorExtractDynamic(               \
        rhs.M.spvm, rhs.idx)) op static_cast<float>(lhs)};                     \
  }
  OP(bool, ==)
  OP(bool, !=)
  OP(bool, <)
  OP(bool, >)
  OP(bool, <=)
  OP(bool, >=)
#undef OP
#else // __SYCL_DEVICE_ONLY__
#define OP(type, op)                                                           \
  friend type operator op(                                                     \
      const wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows,     \
                       NumCols, Use, Layout, Group> &,                         \
      const sycl::ext::oneapi::experimental::bfloat16 &) {                     \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::experimental::bfloat16 &,                       \
      const wi_element<sycl::ext::oneapi::experimental::bfloat16, NumRows,     \
                       NumCols, Use, Layout, Group> &) {                       \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }
  OP(sycl::ext::oneapi::experimental::bfloat16, +)
  OP(sycl::ext::oneapi::experimental::bfloat16, -)
  OP(sycl::ext::oneapi::experimental::bfloat16, *)
  OP(sycl::ext::oneapi::experimental::bfloat16, /)
  OP(bool, ==)
  OP(bool, !=)
  OP(bool, <)
  OP(bool, >)
  OP(bool, <=)
  OP(bool, >=)
#undef OP
#endif // __SYCL_DEVICE_ONLY__
};

template <typename T, size_t NumRows, size_t NumCols, use Use, layout Layout,
          typename Group>
class wi_data {
  joint_matrix<T, Use, NumRows, NumCols, Layout, Group> &M;

public:
  wi_data(joint_matrix<T, Use, NumRows, NumCols, Layout, Group> &Mat)
      : M(Mat) {}
  size_t length() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_JointMatrixWorkItemLengthINTEL(M.spvm);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
  wi_element<T, NumRows, NumCols, Use, Layout, Group> operator[](size_t i) {
    return wi_element<T, NumRows, NumCols, Use, Layout, Group>(M, i);
  }
};

} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
