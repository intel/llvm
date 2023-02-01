//===------- matrix-unified.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once
#include "matrix-intel.hpp"
#include <sycl/ext/oneapi/matrix/matrix-tensorcores.hpp>
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout>
struct joint_matrix {

#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  sycl::ext::oneapi::detail::joint_matrix_cuda<T, Use, Rows, Cols, Layout>
      cuda_impl;
#else
  __spv::__spirv_JointMatrixINTEL<
      T, Rows, Cols, spv_matrix_layout_traits<Layout>::value,
      spv_scope_traits<Group>::value, spv_matrix_use_traits<Use>::value> *spvm;
#endif // defined(__NVPTX__)
#endif // defined(__SYCL_DEVICE_ONLY__)

  joint_matrix() {
#ifndef __SYCL_DEVICE_ONLY__
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }
};

#ifdef __SYCL_DEVICE_ONLY__
template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout>
class wi_data {

  joint_matrix<Group, T, Use, Rows, Cols, Layout> &jm;

  wi_data(joint_matrix<Group, T, Use, Rows, Cols, Layout> &_jm) : jm(_jm){};

  template <typename Grp, typename Type, use UseJm, size_t NumRows,
            size_t NumCols, layout LayoutJm>
  friend decltype(auto)
  get_wi_data(Grp,
              joint_matrix<Grp, Type, UseJm, NumRows, NumCols, LayoutJm> &);

public:
  size_t length() {
#if defined(__NVPTX__)
    return jm.cuda_impl.wi_marray.size();
#else
    return __spirv_JointMatrixWorkItemLengthINTEL(jm.spvm);
#endif
  };

  decltype(auto) operator[](size_t i) {
#if defined(__NVPTX__)
    return (jm.cuda_impl.wi_marray[i]);
#else
    return wi_element<T, Rows, Cols, Use, Layout, Group>(jm, i);
#endif
  };
};
#else
template <typename type, size_t size> class wi_data {
  marray<type, size> &data;
  wi_data(marray<type, size> &wi_marray) : data(wi_marray){};
  template <typename Grp, typename Type, use UseJm, size_t NumRows,
            size_t NumCols, layout LayoutJm>
  friend decltype(auto)
  get_wi_data(Grp,
              joint_matrix<Grp, Type, UseJm, NumRows, NumCols, LayoutJm> &);

public:
  size_t length() { return data.size(); };

  type &operator[](size_t i) { return data[i]; };
};
#endif

template <typename Group, typename T, use Use, size_t Rows, size_t Cols,
          layout Layout>
inline __SYCL_ALWAYS_INLINE decltype(auto)
get_wi_data(Group sg, joint_matrix<Group, T, Use, Rows, Cols, Layout> &jm) {
#if defined(__SYCL_DEVICE_ONLY__)
  std::ignore = sg;
  return wi_data(jm);
#else
  std::ignore = sg;
  std::ignore = jm;
  if constexpr (std::is_same_v<T, precision::tf32>) {
    marray<float, 1> unused{};
    return wi_data<float, 1>(unused);
  } else {
    marray<T, 1> unused{};
    return wi_data<T, 1>(unused);
  }
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, size_t NumRows, size_t NumCols, use Use,
          layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group sg,
                  joint_matrix<Group, T, Use, NumRows, NumCols, Layout> &res,
                  const T2 &v) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  res.cuda_impl.wi_marray = v;
#else
  using storage_element_type = typename helper_traits<T>::storage_element_type;
  res.spvm =
      __spirv_CompositeConstruct<storage_element_type, T, NumRows, NumCols,
                                 spv_matrix_use_traits<Use>::value,
                                 spv_matrix_layout_traits<Layout>::value>(
          static_cast<storage_element_type>(v));
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = v;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value, bool> =
        true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg,
    joint_matrix<Group, S, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::load_accumulator_cuda(res.cuda_impl, src, stride,
                                                   Layout);
#else
  T *Ptr = src.get();
  switch (Layout) {
  default:
    assert(false && "Invalid Memory Layout!");
  case layout::row_major:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        T, S, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::col_major:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        T, S, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case sycl::ext::intel::experimental::matrix::layout::packed:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        T, S, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::Packed,
        spv_scope_traits<Group>::value);
    break;
  }
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = src;
  std::ignore = stride;
  std::ignore = Layout;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, use Use, size_t NumRows,
    size_t NumCols, matrix::layout Layout, access::address_space Space,
    access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value ||
                         (std::is_same<S, precision::tf32>::value &&
                          std::is_same<std::remove_const_t<T>, float>::value),
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_load(Group sg,
                  joint_matrix<Group, S, Use, NumRows, NumCols, Layout> &res,
                  multi_ptr<T, Space, IsDecorated> src, size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::load_multiplicand_cuda<S, T, NumRows, NumCols, Use,
                                                    Layout, Space>(
      res.cuda_impl, src, stride);
#else
  T *Ptr = src.get();
  res.spvm =
      __spirv_JointMatrixLoadINTEL<T, S, NumRows, NumCols,
                                   spv_matrix_use_traits<Use>::value,
                                   spv_matrix_layout_traits<Layout>::value>(
          Ptr, stride, spv_matrix_layout_traits<Layout>::value,
          spv_scope_traits<Group>::value);
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = src;
  std::ignore = stride;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group sg,
    joint_matrix<Group, T, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::joint_matrix_store_cuda<T, NumRows, NumCols,
                                                     Space>(src.cuda_impl, dst,
                                                            stride, Layout);
#else
  T *Ptr = dst.get();
  switch (Layout) {
  default:
    assert(false && "Invalid Memory Layout!");
  case layout::row_major:
    __spirv_JointMatrixStoreINTEL<
        T, T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::col_major:
    __spirv_JointMatrixStoreINTEL<
        T, T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case sycl::ext::intel::experimental::matrix::layout::packed:
    __spirv_JointMatrixStoreINTEL<
        T, T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::Packed,
        spv_scope_traits<Group>::value);
    break;
  }
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  std::ignore = Layout;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename Ta, typename Tb, typename Tc, std::size_t M,
          std::size_t K, std::size_t N, layout LayoutA, layout LayoutB>
inline __SYCL_ALWAYS_INLINE
    joint_matrix<Group, Tc, use::accumulator, M, N,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic>
    joint_matrix_mad(
        Group sg, joint_matrix<Group, Ta, use::a, M, K, LayoutA> &A,
        joint_matrix<Group, Tb, use::b, K, N, LayoutB> &B,
        joint_matrix<Group, Tc, use::accumulator, M, N,
                     sycl::ext::oneapi::experimental::matrix::layout::dynamic>
            &C) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  if constexpr (std::is_same<Ta, Tb>::value) {
    joint_matrix<Group, Tc, use::accumulator, M, N,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic>
        D;
    sycl::ext::oneapi::detail::joint_matrix_mad_cuda<Ta, Tc, M, K, N, LayoutA,
                                                     LayoutB>(
        D.cuda_impl, A.cuda_impl, B.cuda_impl, C.cuda_impl);
    return D;
  } else {
    assert(false && "Ta != Tb : In the CUDA backend joint_matrix_mad "
                    "requires that joint_matrix data types Ta and Tb match");
  }
#else
  joint_matrix<Group, Tc, use::accumulator, M, N, layout::dynamic> res;
  if constexpr (std::is_same<Ta, uint16_t>::value &&
                std::is_same<Tb, uint16_t>::value &&
                std::is_same<Tc, float>::value)
    res.spvm = __spirv_JointMatrixMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_unsigned<Ta>::value && std::is_unsigned<Tb>::value)
    res.spvm = __spirv_JointMatrixUUMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_signed<Ta>::value && std::is_unsigned<Tb>::value)
    res.spvm = __spirv_JointMatrixSUMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_unsigned<Ta>::value && std::is_signed<Tb>::value)
    res.spvm = __spirv_JointMatrixUSMadINTEL(A.spvm, B.spvm, C.spvm);
  else
    res.spvm = __spirv_JointMatrixMadINTEL(A.spvm, B.spvm, C.spvm);
  return res;
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = A;
  std::ignore = B;
  std::ignore = C;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

// This function rounds the bottom 13 bits up or down, and then zeros out the
// bottom bits
inline __SYCL_ALWAYS_INLINE float round_to_tf32(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  int32_t tmp_int = __nvvm_f2tf32_rna(a);
  return __nvvm_bitcast_i2f(tmp_int);
#else
  uint32_t tmp_uint = reinterpret_cast<const uint32_t &>(a);
  tmp_uint += 0x1000u;
  tmp_uint &= 0xFFFFE000u;
  float ret = 0;
  std::memcpy(&ret, &tmp_uint, sizeof(float));
  return ret;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
