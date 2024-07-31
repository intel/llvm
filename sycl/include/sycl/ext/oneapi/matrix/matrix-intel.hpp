//==------------------ matrix-intel.hpp - SYCL matrix ----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include "matrix-unified-utils.hpp" // for use, layout, tf32, matrix
#include "utils.hpp"                // for getDecorated

#include <CL/__spirv/spirv_types.hpp>         // for MatrixLayout, MatrixUse
#include <sycl/access/access.hpp>             // for address_space, decorated
#include <sycl/builtins.hpp>                  // for fabs
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE
#include <sycl/exception.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>       // for bfloat16
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp> // for annotated_ptr
#include <sycl/group.hpp>     // for group
#include <sycl/multi_ptr.hpp> // for multi_ptr
#include <sycl/sub_group.hpp> // for sub_group

#include <cstddef>     // for size_t
#include <stdint.h>    // for uint32_t
#include <tuple>       // for ignore, tuple, _Swallo...
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace ext {
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
SPV_MATRIX_LAYOUT_TRAITS(layout::ext_intel_packed, __spv::MatrixLayout::Packed)
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

namespace detail {
// Differentiating between the "element type" and the "storage element type"
template <typename T> struct jm_type_interpretation_helper_trait {
  using element_type = T;
  using storage_element_type = T;
};

template <>
struct jm_type_interpretation_helper_trait<
    sycl::ext::oneapi::experimental::matrix::precision::tf32> {
  using element_type = sycl::ext::oneapi::experimental::matrix::precision::tf32;
  using storage_element_type = float;
};

using namespace sycl::ext::oneapi::experimental::matrix;
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

public:
  using storage_element_type =
      typename oneapi::detail::jm_type_interpretation_helper_trait<
          T>::storage_element_type;
  wi_element(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                 Group, T, Use, NumRows, NumCols, Layout> &Mat,
             std::size_t i)
      : M(Mat), idx(i) {}

  inline __SYCL_ALWAYS_INLINE std::tuple<size_t, size_t> get_coord() {
#if defined(__SYCL_DEVICE_ONLY__)
    __ocl_vec_t<uint32_t, 2> coord =
        __spirv_JointMatrixGetElementCoordINTEL(M.spvm, idx);
    const size_t row = coord[0];
    const size_t col = coord[1];
    return std::make_tuple(row, col);
#else
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  operator storage_element_type() {
#ifdef __SYCL_DEVICE_ONLY__
    storage_element_type elem =
        __spirv_VectorExtractDynamic<storage_element_type, T, NumRows, NumCols,
                                     spv_matrix_use_traits<Use>::value,
                                     spv_matrix_layout_traits<Layout>::value,
                                     spv_scope_traits<Group>::value>(M.spvm,
                                                                     idx);
    return elem;
#else
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  explicit operator bool() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_VectorExtractDynamic<storage_element_type, T, NumRows,
                                        NumCols,
                                        spv_matrix_use_traits<Use>::value,
                                        spv_matrix_layout_traits<Layout>::value,
                                        spv_scope_traits<Group>::value>(
               M.spvm, idx) != static_cast<storage_element_type>(0);
#else
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  template <typename T2> wi_element &operator=(const T2 &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(
        M.spvm, static_cast<storage_element_type>(rhs), idx);
    return *this;
#else
    (void)rhs;
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &
  operator=(const wi_element<T, NumRows, NumCols, Use, Layout, Group> &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(
        M.spvm,
        __spirv_VectorExtractDynamic<storage_element_type, T, NumRows, NumCols,
                                     spv_matrix_use_traits<Use>::value,
                                     spv_matrix_layout_traits<Layout>::value,
                                     spv_scope_traits<Group>::value>(rhs.M.spvm,
                                                                     rhs.idx),
        idx);
    return *this;
#else
    (void)rhs;
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

#if __SYCL_DEVICE_ONLY__
#define OP(op)                                                                 \
  template <typename T2> wi_element &operator op##=(const T2 & rhs) {          \
    M.spvm = __spirv_VectorInsertDynamic(                                      \
        M.spvm,                                                                \
        static_cast<storage_element_type>(                                     \
            __spirv_VectorExtractDynamic<                                      \
                storage_element_type, T, NumRows, NumCols,                     \
                spv_matrix_use_traits<Use>::value,                             \
                spv_matrix_layout_traits<Layout>::value,                       \
                spv_scope_traits<Group>::value>(M.spvm, idx)                   \
                op static_cast<storage_element_type>(rhs)),                    \
        idx);                                                                  \
    return *this;                                                              \
  }
#else // __SYCL_DEVICE_ONLY__
#define OP(op)                                                                 \
  template <typename T2> wi_element &operator op##=(const T2 & rhs) {          \
    (void)rhs;                                                                 \
    throw exception(make_error_code(errc::runtime),                            \
                    "joint matrix is not supported on host.");                 \
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

public:
  wi_element(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                 Group, sycl::ext::oneapi::bfloat16, Use, NumRows, NumCols,
                 Layout> &Mat,
             std::size_t i)
      : M(Mat), idx(i) {}

  inline __SYCL_ALWAYS_INLINE std::tuple<uint32_t, uint32_t> get_coord() {
#if defined(__SYCL_DEVICE_ONLY__)
    __ocl_vec_t<uint32_t, 2> coord =
        __spirv_JointMatrixGetElementCoordINTEL(M.spvm, idx);
    const uint32_t row = coord[0];
    const uint32_t col = coord[1];
    return std::make_tuple(row, col);
#else
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  operator sycl::ext::oneapi::bfloat16() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_VectorExtractDynamic<
        sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, NumRows,
        NumCols, spv_matrix_use_traits<Use>::value,
        spv_matrix_layout_traits<Layout>::value,
        spv_scope_traits<Group>::value>(M.spvm, idx);
#else
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  explicit operator bool() {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fabs(static_cast<float>(
               __spirv_VectorExtractDynamic<
                   sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                   NumRows, NumCols, spv_matrix_use_traits<Use>::value,
                   spv_matrix_layout_traits<Layout>::value,
                   spv_scope_traits<Group>::value>(M.spvm, idx))) >=
           std::numeric_limits<float>::epsilon();
#else
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &operator=(const sycl::ext::oneapi::bfloat16 &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(M.spvm, rhs, idx);
    return *this;
#else
    (void)rhs;
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &operator=(const wi_element<sycl::ext::oneapi::bfloat16, NumRows,
                                         NumCols, Use, Layout, Group> &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    M.spvm = __spirv_VectorInsertDynamic(
        M.spvm,
        __spirv_VectorExtractDynamic<sycl::ext::oneapi::bfloat16,
                                     sycl::ext::oneapi::bfloat16, NumRows,
                                     NumCols, spv_matrix_use_traits<Use>::value,
                                     spv_matrix_layout_traits<Layout>::value,
                                     spv_scope_traits<Group>::value>(rhs.M.spvm,
                                                                     rhs.idx),
        idx);
    return *this;
#else
    (void)rhs;
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

#if __SYCL_DEVICE_ONLY__
#define OP(opassign, op)                                                       \
  wi_element &operator opassign(const sycl::ext::oneapi::bfloat16 & rhs) {     \
    M.spvm = __spirv_VectorInsertDynamic(                                      \
        M.spvm,                                                                \
        __spirv_VectorExtractDynamic<                                          \
            sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, NumRows, \
            NumCols, spv_matrix_use_traits<Use>::value,                        \
            spv_matrix_layout_traits<Layout>::value,                           \
            spv_scope_traits<Group>::value>(M.spvm, idx) op rhs,               \
        idx);                                                                  \
    return *this;                                                              \
  }
#else // __SYCL_DEVICE_ONLY__
#define OP(opassign, op)                                                       \
  wi_element &operator opassign(const sycl::ext::oneapi::bfloat16 & rhs) {     \
    (void)rhs;                                                                 \
    throw exception(make_error_code(errc::runtime),                            \
                    "joint matrix is not supported on host.");                 \
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
    return __spirv_VectorExtractDynamic<                                       \
        sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, NumRows,     \
        NumCols, spv_matrix_use_traits<Use>::value,                            \
        spv_matrix_layout_traits<Layout>::value,                               \
        spv_scope_traits<Group>::value>(lhs.M.spvm, lhs.idx) op rhs;           \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &lhs,                                  \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &rhs) {                                  \
    return __spirv_VectorExtractDynamic<                                       \
        sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, NumRows,     \
        NumCols, spv_matrix_use_traits<Use>::value,                            \
        spv_matrix_layout_traits<Layout>::value,                               \
        spv_scope_traits<Group>::value>(rhs.M.spvm, rhs.idx) op lhs;           \
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
    return type{static_cast<float>(                                            \
        __spirv_VectorExtractDynamic<                                          \
            sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, NumRows, \
            NumCols, spv_matrix_use_traits<Use>::value,                        \
            spv_matrix_layout_traits<Layout>::value,                           \
            spv_scope_traits<Group>::value>(lhs.M.spvm, lhs.idx))              \
                    op static_cast<float>(rhs)};                               \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &lhs,                                  \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &rhs) {                                  \
    return type{static_cast<float>(                                            \
        __spirv_VectorExtractDynamic<                                          \
            sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, NumRows, \
            NumCols, spv_matrix_use_traits<Use>::value,                        \
            spv_matrix_layout_traits<Layout>::value,                           \
            spv_scope_traits<Group>::value>(rhs.M.spvm, rhs.idx))              \
                    op static_cast<float>(lhs)};                               \
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
    throw exception(make_error_code(errc::runtime),                            \
                    "joint matrix is not supported on host.");                 \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &,                                     \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Use,     \
                       Layout, Group> &) {                                     \
    throw exception(make_error_code(errc::runtime),                            \
                    "joint matrix is not supported on host.");                 \
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
    throw exception(make_error_code(errc::runtime),
                    "joint matrix is not supported on host.");
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
} // namespace detail
} // namespace oneapi

namespace intel::experimental::matrix {
template <
    typename Group, typename T, typename Tp,
    sycl::ext::oneapi::experimental::matrix::use Use, size_t NumRows,
    size_t NumCols, sycl::ext::oneapi::experimental::matrix::layout Layout,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<Use == sycl::ext::oneapi::experimental::matrix::use::a ||
                         Use == sycl::ext::oneapi::experimental::matrix::use::b,
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_store(Group,
                   const sycl::ext::oneapi::experimental::matrix::joint_matrix<
                       Group, Tp, Use, NumRows, NumCols, Layout> &src,
                   multi_ptr<T, Space, IsDecorated> dst, size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support store to private memory!");
#if defined(__NVPTX__)
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw exception(
      make_error_code(errc::runtime),
      "This version of the matrix extension is only currently supported on "
      "intel devices");
#else
  // intel's impl
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(dst);
  __spirv_JointMatrixStoreINTEL<DecorT, Tp, NumRows, NumCols,
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
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename T, typename Tp,
    sycl::ext::oneapi::experimental::matrix::use Use, size_t NumRows,
    size_t NumCols, sycl::ext::oneapi::experimental::matrix::layout Layout,
    typename PropertyListT,
    std::enable_if_t<Use == sycl::ext::oneapi::experimental::matrix::use::a ||
                         Use == sycl::ext::oneapi::experimental::matrix::use::b,
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group,
    const sycl::ext::oneapi::experimental::matrix::joint_matrix<
        Group, Tp, Use, NumRows, NumCols, Layout> &src,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> dst,
    size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw exception(
      make_error_code(errc::runtime),
      "This version of the matrix extension is only currently supported on "
      "intel devices");
#else
  // intel's impl
  T *Ptr = dst.get();
  __spirv_JointMatrixStoreINTEL<T, Tp, NumRows, NumCols,
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
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T,
          sycl::ext::oneapi::experimental::matrix::use Use, size_t Rows,
          size_t Cols, sycl::ext::oneapi::experimental::matrix::layout Layout,
          typename F>
inline __SYCL_ALWAYS_INLINE void joint_matrix_apply(
    Group sg,
    sycl::ext::oneapi::experimental::matrix::joint_matrix<Group, T, Use, Rows,
                                                          Cols, Layout> &jm,
    F &&lambda) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  for (int i = 0; i < jm.matrix_impl.wi_marray.size(); i++) {
    lambda(jm.matrix_impl.wi_marray[i]);
  }
#else // NVPTX
  using storage_element_type =
      typename oneapi::detail::jm_type_interpretation_helper_trait<
          T>::storage_element_type;
  auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, jm);
  for (int i = 0; i < wi_data_c.length(); i++) {
    storage_element_type element = wi_data_c[i];
    auto [row, col] = wi_data_c[i].get_coord();
    lambda(element, row, col);
    wi_data_c[i] = element;
  }
#endif
#else
  std::ignore = sg;
  std::ignore = jm;
  std::ignore = lambda;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif
}

using namespace sycl::ext::oneapi::experimental::matrix;

// Begin out-of-bounds API

template <typename Group, typename T, size_t NumRows, size_t NumCols, use Use,
          layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void joint_matrix_fill_checked(
    Group, joint_matrix<Group, T, Use, NumRows, NumCols, Layout> &Res,
    const T2 &Value, size_t Height, size_t Width, size_t CoordX,
    size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  using storage_element_type =
      typename oneapi::detail::jm_type_interpretation_helper_trait<
          T>::storage_element_type;
  Res.spvm = __spirv_CooperativeMatrixConstructCheckedINTEL<
      storage_element_type, T, NumRows, NumCols,
      spv_matrix_use_traits<Use>::value,
      spv_matrix_layout_traits<Layout>::value>(
      CoordX, CoordY, Height, Width, static_cast<storage_element_type>(Value));
#else
  std::ignore = Res;
  std::ignore = Value;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value, bool> =
        true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load_checked(
    Group sg,
    joint_matrix<Group, S, use::accumulator, NumRows, NumCols, layout::dynamic>
        &Res,
    multi_ptr<T, Space, IsDecorated> Src, size_t Stride, layout Layout,
    size_t Height, size_t Width, size_t CoordX, size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support load from private memory!");
  std::ignore = sg;
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(Src);
  Res.spvm = __spirv_CooperativeMatrixLoadCheckedINTEL<
      DecorT, S, NumRows, NumCols,
      spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, CoordX, CoordY, sycl::detail::joint_matrix_layout_to_spv(Layout),
      Height, Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Res;
  std::ignore = Src;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = Layout;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, use Use, size_t NumRows,
    size_t NumCols, layout Layout, access::address_space Space,
    access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value ||
                         (std::is_same<S, precision::tf32>::value &&
                          std::is_same<std::remove_const_t<T>, float>::value),
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load_checked(
    Group sg, joint_matrix<Group, S, Use, NumRows, NumCols, Layout> &Res,
    multi_ptr<T, Space, IsDecorated> Src, size_t Stride, size_t Height,
    size_t Width, size_t CoordX, size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support load from private memory!");
  std::ignore = sg;
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(Src);
  Res.spvm = __spirv_CooperativeMatrixLoadCheckedINTEL<
      DecorT, S, NumRows, NumCols, spv_matrix_use_traits<Use>::value,
      spv_matrix_layout_traits<Layout>::value>(
      Ptr, CoordX, CoordY, spv_matrix_layout_traits<Layout>::value, Height,
      Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Res;
  std::ignore = Src;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store_checked(
    Group sg,
    joint_matrix<Group, T, use::accumulator, NumRows, NumCols, layout::dynamic>
        &Src,
    multi_ptr<T, Space, IsDecorated> Dst, size_t Stride, layout Layout,
    size_t Height, size_t Width, size_t CoordX, size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support store to private memory!");
  std::ignore = sg;
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(Dst);
  __spirv_CooperativeMatrixStoreCheckedINTEL<
      DecorT, T, NumRows, NumCols,
      spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, CoordX, CoordY, Src.spvm,
      sycl::detail::joint_matrix_layout_to_spv(Layout), Height, Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Src;
  std::ignore = Dst;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = Layout;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, typename Tp, use Use, size_t NumRows,
          size_t NumCols, layout Layout, access::address_space Space,
          access::decorated IsDecorated,
          std::enable_if_t<Use == use::a || Use == use::b, bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store_checked(
    Group sg, const joint_matrix<Group, Tp, Use, NumRows, NumCols, Layout> &Src,
    multi_ptr<T, Space, IsDecorated> Dst, size_t Stride, size_t Height,
    size_t Width, size_t CoordX, size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support store to private memory!");
  std::ignore = sg;
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(Dst);
  __spirv_CooperativeMatrixStoreCheckedINTEL<
      DecorT, Tp, NumRows, NumCols, spv_matrix_use_traits<Use>::value,
      spv_matrix_layout_traits<Layout>::value>(
      Ptr, CoordX, CoordY, Src.spvm, spv_matrix_layout_traits<Layout>::value,
      Height, Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Src;
  std::ignore = Dst;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

// Annotated pointer overloads:
template <typename Group, typename S, typename T, size_t NumRows,
          size_t NumCols, typename PropertyListT,
          std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value,
                           bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load_checked(
    Group sg,
    joint_matrix<Group, S, use::accumulator, NumRows, NumCols, layout::dynamic>
        &Res,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> Src,
    size_t Stride, layout Layout, size_t Height, size_t Width, size_t CoordX,
    size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  std::ignore = sg;
  T *Ptr = Src.get();
  Res.spvm = __spirv_CooperativeMatrixLoadCheckedINTEL<
      T, S, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, CoordX, CoordY, sycl::detail::joint_matrix_layout_to_spv(Layout),
      Height, Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Res;
  std::ignore = Src;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = Layout;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, use Use, size_t NumRows,
    size_t NumCols, layout Layout, typename PropertyListT,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value ||
                         (std::is_same<S, precision::tf32>::value &&
                          std::is_same<std::remove_const_t<T>, float>::value),
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load_checked(
    Group sg, joint_matrix<Group, S, Use, NumRows, NumCols, Layout> &Res,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> Src,
    size_t Stride, size_t Height, size_t Width, size_t CoordX, size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  std::ignore = sg;
  T *Ptr = Src.get();
  Res.spvm = __spirv_CooperativeMatrixLoadCheckedINTEL<
      T, S, NumRows, NumCols, spv_matrix_use_traits<Use>::value,
      spv_matrix_layout_traits<Layout>::value>(
      Ptr, CoordX, CoordY, spv_matrix_layout_traits<Layout>::value, Height,
      Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Res;
  std::ignore = Src;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          typename PropertyListT>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store_checked(
    Group sg,
    joint_matrix<Group, T, use::accumulator, NumRows, NumCols, layout::dynamic>
        &Src,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> Dst,
    size_t Stride, layout Layout, size_t Height, size_t Width, size_t CoordX,
    size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  std::ignore = sg;
  T *Ptr = Dst.get();
  __spirv_CooperativeMatrixStoreCheckedINTEL<
      T, T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, CoordX, CoordY, Src.spvm,
      sycl::detail::joint_matrix_layout_to_spv(Layout), Height, Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Src;
  std::ignore = Dst;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = Layout;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, typename Tp, use Use, size_t NumRows,
          size_t NumCols, layout Layout, typename PropertyListT,
          std::enable_if_t<Use == use::a || Use == use::b, bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store_checked(
    Group sg, const joint_matrix<Group, Tp, Use, NumRows, NumCols, Layout> &Src,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> Dst,
    size_t Stride, size_t Height, size_t Width, size_t CoordX, size_t CoordY) {
#if defined(__SYCL_DEVICE_ONLY__)
  std::ignore = sg;
  T *Ptr = Dst.get();
  __spirv_CooperativeMatrixStoreCheckedINTEL<
      T, Tp, NumRows, NumCols, spv_matrix_use_traits<Use>::value,
      spv_matrix_layout_traits<Layout>::value>(
      Ptr, CoordX, CoordY, Src.spvm, spv_matrix_layout_traits<Layout>::value,
      Height, Width, Stride);
#else
  std::ignore = sg;
  std::ignore = Src;
  std::ignore = Dst;
  std::ignore = Stride;
  std::ignore = Height;
  std::ignore = Width;
  std::ignore = CoordX;
  std::ignore = CoordY;
  throw exception(make_error_code(errc::runtime),
                  "joint matrix is not supported on host.");
#endif // defined(__SYCL_DEVICE_ONLY__)
}
// End out-of-bounds API

} // namespace intel::experimental::matrix

} // namespace ext
} // namespace _V1
} // namespace sycl
