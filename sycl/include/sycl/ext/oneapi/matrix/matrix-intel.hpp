//==------------------ matrix-intel.hpp - SYCL matrix ----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include "matrix-unified-utils.hpp"
#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/feature_test.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel::experimental::matrix::layout {
constexpr sycl::ext::oneapi::experimental::matrix::layout packed =
    static_cast<sycl::ext::oneapi::experimental::matrix::layout>(2);
}
namespace oneapi {
namespace experimental {
namespace matrix {

template <layout Layout> struct spv_matrix_layout_traits {
  static constexpr __spv::MatrixLayout value = __spv::MatrixLayout::Dynamic;
};

#define SPV_MATRIX_LAYOUT_TRAITS(LAYOUT, SPV_LAYOUT)                           \
  template <> struct spv_matrix_layout_traits<LAYOUT> {                        \
    static constexpr __spv::MatrixLayout value = SPV_LAYOUT;                   \
  };

SPV_MATRIX_LAYOUT_TRAITS(layout::row_major, __spv::MatrixLayout::RowMajor)
SPV_MATRIX_LAYOUT_TRAITS(layout::col_major, __spv::MatrixLayout::ColumnMajor)
SPV_MATRIX_LAYOUT_TRAITS(sycl::ext::intel::experimental::matrix::layout::packed,
                         __spv::MatrixLayout::Packed)
SPV_MATRIX_LAYOUT_TRAITS(layout::dynamic, __spv::MatrixLayout::Dynamic)

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
template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout>
struct joint_matrix;

} // namespace matrix
} // namespace experimental
} // namespace oneapi

namespace intel::experimental::matrix {

// Begin wi_element definition

template <typename T, size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::use Use,
          sycl::ext::oneapi::experimental::matrix::layout Layout =
              sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          typename Group = sycl::sub_group>
class wi_element {
  sycl::ext::oneapi::experimental::matrix::joint_matrix<Group, T, Use, NumRows,
                                                        NumCols, Layout> &M;
  std::size_t idx;

  template <typename T1, size_t NRows, size_t NCols,
            sycl::ext::oneapi::experimental::matrix::use Use1,
            sycl::ext::oneapi::experimental::matrix::layout Layout1,
            typename Grp>
  friend std::tuple<uint32_t, uint32_t>
  get_coord(wi_element<T1, NRows, NCols, Use1, Layout1, Grp> &);

public:
  wi_element(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                 Group, T, Use, NumRows, NumCols, Layout> &Mat,
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

template <size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::use Use,
          sycl::ext::oneapi::experimental::matrix::layout Layout,
          typename Group>
class wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use, Layout,
                 Group> {
  sycl::ext::oneapi::experimental::matrix::joint_matrix<
      Group, sycl::ext::oneapi::bfloat16, Use, NumRows, NumCols, Layout> &M;
  std::size_t idx;

  template <typename T1, size_t NRows, size_t NCols,
            sycl::ext::oneapi::experimental::matrix::use Use1,
            sycl::ext::oneapi::experimental::matrix::layout Layout1,
            typename Grp>
  friend std::tuple<uint32_t, uint32_t>
  get_coord(wi_element<T1, NRows, NCols, Use1, Layout1, Grp> &);

public:
  wi_element(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                 Group, sycl::ext::oneapi::bfloat16, Use, NumRows, NumCols,
                 Layout> &Mat,
             std::size_t i)
      : M(Mat), idx(i) {}
  operator sycl::ext::oneapi::bfloat16() {
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

  wi_element &operator=(const sycl::ext::oneapi::bfloat16 &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(M.spvm, rhs, idx);
    return *this;
#else
    (void)rhs;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &operator=(const wi_element<sycl::ext::oneapi::bfloat16, NumRows,
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
  wi_element &operator opassign(const sycl::ext::oneapi::bfloat16 &rhs) {      \
    M.spvm = __spirv_VectorInsertDynamic(                                      \
        M.spvm, __spirv_VectorExtractDynamic(M.spvm, idx) op rhs, idx);        \
    return *this;                                                              \
  }
#else // __SYCL_DEVICE_ONLY__
#define OP(opassign, op)                                                       \
  wi_element &operator opassign(const sycl::ext::oneapi::bfloat16 &rhs) {      \
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
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &lhs,                                    \
      const sycl::ext::oneapi::bfloat16 &rhs) {                                \
    return __spirv_VectorExtractDynamic(lhs.M.spvm, lhs.idx) op rhs;           \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &lhs,                                  \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &rhs) {                                  \
    return __spirv_VectorExtractDynamic(rhs.M.spvm, rhs.idx) op lhs;           \
  }
  OP(sycl::ext::oneapi::bfloat16, +)
  OP(sycl::ext::oneapi::bfloat16, -)
  OP(sycl::ext::oneapi::bfloat16, *)
  OP(sycl::ext::oneapi::bfloat16, /)
#undef OP
#define OP(type, op)                                                           \
  friend type operator op(                                                     \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &lhs,                                    \
      const sycl::ext::oneapi::bfloat16 &rhs) {                                \
    return type{static_cast<float>(__spirv_VectorExtractDynamic(               \
        lhs.M.spvm, lhs.idx)) op static_cast<float>(rhs)};                     \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &lhs,                                  \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &rhs) {                                  \
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
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &,                                       \
      const sycl::ext::oneapi::bfloat16 &) {                                   \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &,                                     \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &) {                                     \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }
  OP(sycl::ext::oneapi::bfloat16, +)
  OP(sycl::ext::oneapi::bfloat16, -)
  OP(sycl::ext::oneapi::bfloat16, *)
  OP(sycl::ext::oneapi::bfloat16, /)
  OP(bool, ==)
  OP(bool, !=)
  OP(bool, <)
  OP(bool, >)
  OP(bool, <=)
  OP(bool, >=)
#undef OP
#endif // __SYCL_DEVICE_ONLY__
};

// End wi_element definition

template <typename T, size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::use Use,
          sycl::ext::oneapi::experimental::matrix::layout Layout,
          typename Group>
inline __SYCL_ALWAYS_INLINE std::tuple<uint32_t, uint32_t>
get_coord(wi_element<T, NumRows, NumCols, Use, Layout, Group> &we) {
#if defined(__SYCL_DEVICE_ONLY__)
  __ocl_vec_t<uint32_t, 2> coord =
      __spirv_JointMatrixGetElementCoordINTEL(we.M.spvm, we.idx);
  const uint32_t row = coord[0];
  const uint32_t col = coord[1];
  return std::make_tuple(row, col);
#else
  std::ignore = we;
  throw runtime_error(
      "get_coord is only supported on Intel XMX and AMX devices.",
      PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}

// To make host compilation possible, here the argument is not a wi_element
// type, but just base data types e.g. float, int8 etc.
template <typename T>
inline __SYCL_ALWAYS_INLINE std::tuple<uint32_t, uint32_t> get_coord(T &we) {
  std::ignore = we;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
}

// Begin wi_data definition

template <typename Group, typename T,
          sycl::ext::oneapi::experimental::matrix::use Use, size_t Rows,
          size_t Cols, sycl::ext::oneapi::experimental::matrix::layout Layout>
class wi_data {

  sycl::ext::oneapi::experimental::matrix::joint_matrix<Group, T, Use, Rows,
                                                        Cols, Layout> &jm;

  wi_data(sycl::ext::oneapi::experimental::matrix::joint_matrix<
          Group, T, Use, Rows, Cols, Layout> &_jm)
      : jm(_jm){};

  template <typename Grp, typename Type,
            sycl::ext::oneapi::experimental::matrix::use UseJm, size_t NumRows,
            size_t NumCols,
            sycl::ext::oneapi::experimental::matrix::layout LayoutJm>
  friend decltype(auto)
  get_wi_data(Grp, sycl::ext::oneapi::experimental::matrix::joint_matrix<
                       Grp, Type, UseJm, NumRows, NumCols, LayoutJm> &);

public:
  size_t length() {
#if __SYCL_DEVICE_ONLY__
    return __spirv_JointMatrixWorkItemLengthINTEL(jm.spvm);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  };

  decltype(auto) operator[](size_t i) {
    return wi_element<T, Rows, Cols, Use, Layout, Group>(jm, i);
  };
};

template <typename Group, typename T,
          sycl::ext::oneapi::experimental::matrix::use Use, size_t Rows,
          size_t Cols, sycl::ext::oneapi::experimental::matrix::layout Layout>
inline __SYCL_ALWAYS_INLINE decltype(auto)
get_wi_data(Group sg, sycl::ext::oneapi::experimental::matrix::joint_matrix<
                          Group, T, Use, Rows, Cols, Layout> &jm) {
  std::ignore = sg;
  return wi_data(jm);
}

// End wi_data definition

template <
    typename Group, typename T,
    sycl::ext::oneapi::experimental::matrix::use Use, size_t NumRows,
    size_t NumCols, sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<Use == sycl::ext::oneapi::experimental::matrix::use::a ||
                         Use == sycl::ext::oneapi::experimental::matrix::use::b,
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_store(Group sg,
                   sycl::ext::oneapi::experimental::matrix::joint_matrix<
                       Group, T, Use, NumRows, NumCols, Layout> &src,
                   multi_ptr<T, Space, IsDecorated> dst, size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw runtime_error(
      "This version of the matrix extension is only currently supported on "
      "intel devices",
      PI_ERROR_INVALID_DEVICE);
#else
  // intel's impl
  T *Ptr = dst.get();
  __spirv_JointMatrixStoreINTEL<T, NumRows, NumCols,
                                sycl::ext::oneapi::experimental::matrix::
                                    spv_matrix_use_traits<Use>::value,
                                sycl::ext::oneapi::experimental::matrix::
                                    spv_matrix_layout_traits<Layout>::value>(
      Ptr, src.spvm, stride,
      sycl::ext::oneapi::experimental::matrix::spv_matrix_layout_traits<
          Layout>::value,
      sycl::ext::oneapi::experimental::matrix::spv_scope_traits<Group>::value);
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}
} // namespace intel::experimental::matrix

} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
