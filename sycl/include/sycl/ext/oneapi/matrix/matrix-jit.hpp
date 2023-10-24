//==---------------- matrix-jit.hpp - SYCL matrix --------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include "utils.hpp"
#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/feature_test.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::matrix {

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
class wi_data;

template <typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          typename Group = sycl::sub_group>
struct joint_matrix {
public:
  __spv::__spirv_JointMatrixINTEL<
      T, NumRows, NumCols, spv_matrix_layout_traits<Layout>::value,
      spv_scope_traits<Group>::value> *spvm;
  joint_matrix(Group sg) {
#ifndef __SYCL_DEVICE_ONLY__
    (void)sg;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  inline __SYCL_ALWAYS_INLINE wi_data<T, NumRows, NumCols, Layout, Group>
  get_wi_data() {
    return wi_data<T, NumRows, NumCols, Layout, Group>(*this);
  }

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__)
  // Generate a non-trivial assignment operator and copy c'tor that prevents
  // memcpy from being generated.
  // TODO: to remove, when either IGC can handle alloca JointMatrix or
  // combination of InstCombine + SROA + mem2reg can remove it
  joint_matrix(const joint_matrix &other) {
    spvm = other.spvm;
    return *this;
  }

  joint_matrix &operator=(const joint_matrix &rhs) {
    spvm = rhs.spvm;
    return *this;
  }
#endif // defined(__SPIR__)
#endif
};

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout = matrix_layout::row_major,
          access::address_space Space, access::decorated IsDecorated>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg, joint_matrix<T, NumRows, NumCols, Layout, Group> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride, matrix_layout MemL) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support load from private memory!");
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(src);
  switch (MemL) {
  default:
    assert(false && "Invalid Memory Layout!");
  case matrix_layout::row_major:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<DecorT, T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
            Ptr, stride, __spv::MatrixLayout::RowMajor,
            spv_scope_traits<Group>::value);
    break;
  case matrix_layout::col_major:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<DecorT, T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
            Ptr, stride, __spv::MatrixLayout::ColumnMajor,
            spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_a:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<DecorT, T, NumRows, NumCols,
                                     spv_matrix_layout_traits<Layout>::value>(
            Ptr, stride, __spv::MatrixLayout::PackedA,
            spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_b:
    res.spvm =
        __spirv_JointMatrixLoadINTEL<DecorT, T, NumRows, NumCols,
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
                      PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout MatL = matrix_layout::row_major,
          access::address_space Space, access::decorated IsDecorated>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group sg, joint_matrix<T, NumRows, NumCols, MatL, Group> &src,
    multi_ptr<T, Space, IsDecorated> res, size_t stride, matrix_layout MemL) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support store to private memory!");
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(res);
  switch (MemL) {
  default:
    assert(false && "Invalid Memory Layout!");
  case matrix_layout::row_major:
    __spirv_JointMatrixStoreINTEL<DecorT, T, NumRows, NumCols,
                                  spv_matrix_layout_traits<MatL>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case matrix_layout::col_major:
    __spirv_JointMatrixStoreINTEL<DecorT, T, NumRows, NumCols,
                                  spv_matrix_layout_traits<MatL>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_a:
    __spirv_JointMatrixStoreINTEL<DecorT, T, NumRows, NumCols,
                                  spv_matrix_layout_traits<MatL>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::PackedA,
        spv_scope_traits<Group>::value);
    break;
  case matrix_layout::packed_b:
    __spirv_JointMatrixStoreINTEL<DecorT, T, NumRows, NumCols,
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
                      PI_ERROR_INVALID_DEVICE);
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
                      PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group sg,
                  joint_matrix<T, NumRows, NumCols, Layout, Group> &res,
                  const T2 v) {
  // We kept the unused "sg" in joint_matrix_fill to match the other DPC++
  // functions
  (void)sg;
#ifdef __SYCL_DEVICE_ONLY__
  res.spvm =
      __spirv_CompositeConstruct<T, NumRows, NumCols,
                                 spv_matrix_layout_traits<Layout>::value>(
          static_cast<T>(v));

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
  operator=(const wi_element<T, NumRows, NumCols, Layout, Group> &rhs) {
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

// Note that similarly to the other matrix functions, uint16_t is used here to
// represent bf16 type. Since the AMX and DPAS implementations don't support
// uint16_t, this interpretation is possible. This design choice was made before
// the introduction of SYCL experimental bfloat16 type. Our plan is to move
// towards using the SYCL bfloat16. But since it is still experimental, we will
// probably keep both uint16 interpretation and SYCL bfloat16.
template <size_t NumRows, size_t NumCols, matrix_layout Layout, typename Group>
class wi_element<uint16_t, NumRows, NumCols, Layout, Group> {
  joint_matrix<uint16_t, NumRows, NumCols, Layout, Group> &M;
  std::size_t idx;

public:
  wi_element(joint_matrix<uint16_t, NumRows, NumCols, Layout, Group> &Mat,
             std::size_t i)
      : M(Mat), idx(i) {}
  operator uint16_t() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_VectorExtractDynamic(M.spvm, idx);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  explicit operator bool() {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fabs(make_fp32(__spirv_VectorExtractDynamic(M.spvm, idx))) >=
           std::numeric_limits<float>::epsilon();
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }

  wi_element &operator=(const uint16_t &rhs) {
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
  operator=(const wi_element<uint16_t, NumRows, NumCols, Layout, Group> &rhs) {
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

  // We use here the following functions for conversion (bf16=>fp32 and
  // fp32=>bf16). This is a workaround until we are able to use
  // __spirv_ConvertFToBF16INTEL and __spirv_ConvertBF16ToFINTEL once these are
  // supported in the CPU backend
  static float make_fp32(uint16_t x) {
    unsigned int y = x;
    y = y << 16;
    float *res = reinterpret_cast<float *>(&y);
    return *res;
  }

  static uint16_t make_bf16(float x) {
    int *res = reinterpret_cast<int *>(&x);
    *res = *res >> 16;
    return (uint16_t)*res;
  }

#if __SYCL_DEVICE_ONLY__
#define OP(op)                                                                 \
  wi_element &operator op##=(const uint16_t &rhs) {                            \
    M.spvm = __spirv_VectorInsertDynamic(                                      \
        M.spvm,                                                                \
        make_bf16(make_fp32(__spirv_VectorExtractDynamic(M.spvm, idx)          \
                                op make_fp32(rhs))),                           \
        idx);                                                                  \
    return *this;                                                              \
  }
#else // __SYCL_DEVICE_ONLY__
#define OP(op)                                                                 \
  wi_element &operator op##=(const uint16_t &rhs) {                            \
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

  template <typename T1, typename T2> struct Converter {
    static T2 convert(const T1 &from) { return static_cast<T2>(from); }
  };

  template <typename T> struct Converter<T, uint16_t> {
    static uint16_t convert(const T &from) { return make_bf16(from); }
  };
#if __SYCL_DEVICE_ONLY__
#define OP(input_type, type, op)                                               \
  friend type operator op(                                                     \
      const wi_element<uint16_t, NumRows, NumCols, Layout, Group> &lhs,        \
      const uint16_t &rhs) {                                                   \
    return Converter<input_type, type>::convert(make_fp32(                     \
        __spirv_VectorExtractDynamic(lhs.M.spvm, lhs.idx)) op make_fp32(rhs)); \
  }                                                                            \
  friend type operator op(                                                     \
      const uint16_t &lhs,                                                     \
      const wi_element<uint16_t, NumRows, NumCols, Layout, Group> &rhs) {      \
    return Converter<input_type, type>::convert(make_fp32(                     \
        __spirv_VectorExtractDynamic(rhs.M.spvm, rhs.idx)) op make_fp32(lhs)); \
  }
#else // __SYCL_DEVICE_ONLY__
#define OP(input_type, type, op)                                               \
  friend type operator op(                                                     \
      const wi_element<uint16_t, NumRows, NumCols, Layout, Group> &lhs,        \
      const uint16_t &rhs) {                                                   \
    (void)lhs;                                                                 \
    (void)rhs;                                                                 \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }                                                                            \
  friend type operator op(                                                     \
      const uint16_t &lhs,                                                     \
      const wi_element<uint16_t, NumRows, NumCols, Layout, Group> &rhs) {      \
    (void)lhs;                                                                 \
    (void)rhs;                                                                 \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }
#endif // __SYCL_DEVICE_ONLY__
  OP(float, uint16_t, +)
  OP(float, uint16_t, -)
  OP(float, uint16_t, *)
  OP(float, uint16_t, /)
  OP(bool, bool, ==)
  OP(bool, bool, !=)
  OP(bool, bool, <)
  OP(bool, bool, >)
  OP(bool, bool, <=)
  OP(bool, bool, >=)
#undef OP
};

template <size_t NumRows, size_t NumCols, matrix_layout Layout, typename Group>
class wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout, Group> {
  joint_matrix<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout, Group> &M;
  std::size_t idx;

public:
  wi_element(joint_matrix<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout,
                          Group> &Mat,
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
    return sycl::fabs(static_cast<float>(__spirv_VectorExtractDynamic(
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
                                         NumCols, Layout, Group> &rhs) {
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
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout,  \
                       Group> &lhs,                                            \
      const sycl::ext::oneapi::bfloat16 &rhs) {                                \
    return __spirv_VectorExtractDynamic(lhs.M.spvm, lhs.idx) op rhs;           \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &lhs,                                  \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout,  \
                       Group> &rhs) {                                          \
    return __spirv_VectorExtractDynamic(rhs.M.spvm, rhs.idx) op lhs;           \
  }
  OP(sycl::ext::oneapi::bfloat16, +)
  OP(sycl::ext::oneapi::bfloat16, -)
  OP(sycl::ext::oneapi::bfloat16, *)
  OP(sycl::ext::oneapi::bfloat16, /)
#undef OP
#define OP(type, op)                                                           \
  friend type operator op(                                                     \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout,  \
                       Group> &lhs,                                            \
      const sycl::ext::oneapi::bfloat16 &rhs) {                                \
    return type{static_cast<float>(__spirv_VectorExtractDynamic(               \
        lhs.M.spvm, lhs.idx)) op static_cast<float>(rhs)};                     \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &lhs,                                  \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout,  \
                       Group> &rhs) {                                          \
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
  friend type operator op(const wi_element<sycl::ext::oneapi::bfloat16,        \
                                           NumRows, NumCols, Layout, Group> &, \
                          const sycl::ext::oneapi::bfloat16 &) {               \
    throw runtime_error("joint matrix is not supported on host device.",       \
                        PI_ERROR_INVALID_DEVICE);                              \
  }                                                                            \
  friend type operator op(                                                     \
      const sycl::ext::oneapi::bfloat16 &,                                     \
      const wi_element<sycl::ext::oneapi::bfloat16, NumRows, NumCols, Layout,  \
                       Group> &) {                                             \
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

template <typename T, size_t NumRows, size_t NumCols, matrix_layout Layout,
          typename Group>
class wi_data {
  joint_matrix<T, NumRows, NumCols, Layout, Group> &M;

public:
  wi_data(joint_matrix<T, NumRows, NumCols, Layout, Group> &Mat) : M(Mat) {}
  size_t length() {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_JointMatrixWorkItemLengthINTEL(M.spvm);
#else
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
  wi_element<T, NumRows, NumCols, Layout, Group> operator[](size_t i) {
    return wi_element<T, NumRows, NumCols, Layout, Group>(M, i);
  }
};

#undef SPV_MATRIX_LAYOUT_TRAITS

} // namespace ext::oneapi::experimental::matrix
} // namespace _V1
} // namespace sycl
