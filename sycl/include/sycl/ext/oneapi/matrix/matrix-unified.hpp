//===------- matrix-unified.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once
#include <sycl/ext/oneapi/matrix/matrix-tensorcores.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

// forward declarations
template <typename T, use Use, size_t Rows, size_t Cols,
          layout Layout = layout::dynamic, typename Group = sycl::sub_group>
struct joint_matrix;

template <typename Group, typename T, size_t NumRows, size_t NumCols, use Use,
          layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group sg,
                  joint_matrix<T, Use, NumRows, NumCols, Layout, Group> &res,
                  const T2 &v);

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
                  joint_matrix<S, Use, NumRows, NumCols, Layout, Group> &res,
                  multi_ptr<T, Space, IsDecorated> src, size_t stride);

template <
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value, bool> =
        true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg,
    joint_matrix<S, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                 Group> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout);

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group sg,
    joint_matrix<T, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                 Group> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout);

template <typename Group, typename Ta, typename Tb, typename Tc, std::size_t M,
          std::size_t K, std::size_t N, layout LayoutA, layout LayoutB>
inline __SYCL_ALWAYS_INLINE
    joint_matrix<Tc, use::accumulator, M, N,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                 Group>
    joint_matrix_mad(
        Group sg, joint_matrix<Ta, use::a, M, K, LayoutA, Group> &A,
        joint_matrix<Tb, use::b, K, N, LayoutB, Group> &B,
        joint_matrix<Tc, use::accumulator, M, N,
                     sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                     Group> &C);

template <typename T, use Use, size_t Rows, size_t Cols, layout Layout,
          typename Group>
struct joint_matrix {

private:
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  sycl::ext::oneapi::detail::joint_matrix_cuda<T, Use, Rows, Cols, Layout,
                                               Group>
      cuda_impl;
#endif
#endif

  template <typename Grp, typename T2, size_t nRows, size_t nCols, use UseT,
            layout LayoutT, typename Type>
  friend inline __SYCL_ALWAYS_INLINE void
  joint_matrix_fill(Grp, joint_matrix<T2, UseT, nRows, nCols, LayoutT, Grp> &,
                    const Type &);

  template <typename Grp, typename jmType, typename Type, use UseL,
            size_t nRows, size_t nCols, matrix::layout LayoutT,
            access::address_space Sp, access::decorated IsDecorated,
            std::enable_if_t<
                std::is_same<jmType, std::remove_const_t<Type>>::value ||
                    (std::is_same<jmType, precision::tf32>::value &&
                     std::is_same<std::remove_const_t<Type>, float>::value),
                bool>>
  friend inline __SYCL_ALWAYS_INLINE void
  joint_matrix_load(Grp,
                    joint_matrix<jmType, UseL, nRows, nCols, LayoutT, Grp> &,
                    multi_ptr<Type, Sp, IsDecorated>, size_t);

  template <typename Grp, typename jmType, typename Type, size_t nRows,
            size_t nCols, access::address_space Sp,
            access::decorated IsDecorated,
            std::enable_if_t<
                std::is_same<jmType, std::remove_const_t<Type>>::value, bool>>
  friend inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
      Grp,
      joint_matrix<jmType, use::accumulator, nRows, nCols,
                   sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                   Grp> &,
      multi_ptr<Type, Sp, IsDecorated>, size_t,
      sycl::ext::oneapi::experimental::matrix::layout);

  template <typename Grp, typename Type, size_t nRows, size_t nCols,
            access::address_space Sp, access::decorated IsDecorated>
  friend inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
      Grp,
      joint_matrix<Type, use::accumulator, nRows, nCols,
                   sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                   Grp> &,
      multi_ptr<Type, Sp, IsDecorated>, size_t,
      sycl::ext::oneapi::experimental::matrix::layout);

  template <typename Grp, typename T_a, typename T_b, typename T_c,
            std::size_t m, std::size_t k, std::size_t n, layout LayoutA,
            layout LayoutB>
  friend inline __SYCL_ALWAYS_INLINE
      joint_matrix<T_c, use::accumulator, m, n,
                   sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                   Grp>
      joint_matrix_mad(
          Grp, joint_matrix<T_a, use::a, m, k, LayoutA, Grp> &,
          joint_matrix<T_b, use::b, k, n, LayoutB, Grp> &,
          joint_matrix<T_c, use::accumulator, m, n,
                       sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                       Grp> &);

public:
  joint_matrix() {
#ifndef __SYCL_DEVICE_ONLY__
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }


  inline __SYCL_ALWAYS_INLINE decltype(auto) get_wi_data() {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
    return wi_data(cuda_impl.wi_marray);
#else
//  intel impl: return wi_data<T, NumRows, NumCols, Use, Layout, Group>(*this);
#endif
#else
  // Host version of get_wi_data required by compiler even though it will never
  // be called because joint_matrix cannot be constructed on host.
    if constexpr (std::is_same_v<T, precision::tf32>) {
      marray<float, 1> unused{};
      return wi_data<float, 1>(unused);
    } else {
      marray<T, 1> unused{};
      return wi_data<T, 1>(unused);
    }
#endif
  };

  // get_wi_marray is only defined for the NVPTX backend.
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  inline __SYCL_ALWAYS_INLINE auto get_wi_marray()
      -> decltype(cuda_impl.wi_marray) & {
    return cuda_impl.wi_marray;
  };
#endif
#else
  // Host version of get_wi_marray required by compiler even though it will
  // never be called because joint_matrix cannot be constructed on host.
  decltype(auto) inline __SYCL_ALWAYS_INLINE get_wi_marray() {
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);

    if constexpr (std::is_same_v<T, precision::tf32>) {
      return marray<float, 1>{};
    } else {
      return marray<T, 1>{};
    }
  };
#endif
};

template <typename Group, typename T, size_t NumRows, size_t NumCols, use Use,
          layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group sg,
                  joint_matrix<T, Use, NumRows, NumCols, Layout, Group> &res,
                  const T2 &v) {
  std::ignore = sg;
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  res.cuda_impl.wi_marray = v;
#endif // defined(__NVPTX__)
#else
  std::ignore = res;
  std::ignore = v;
  throw runtime_error(
      "This version of the matrix extension is only currently supported on "
      "Nvidia devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <
    typename Group, typename S, typename T, size_t NumRows, size_t NumCols,
    access::address_space Space, access::decorated IsDecorated,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value, bool>>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg,
    joint_matrix<S, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                 Group> &res,
    multi_ptr<T, Space, IsDecorated> src, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  sycl::ext::oneapi::detail::load_accumulator_cuda(res.cuda_impl, src, stride,
                                                   Layout);
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = src;
  std::ignore = stride;
  throw runtime_error(
      "This version of the matrix extension is only currently supported on "
      "Nvidia devices",
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
                     bool>>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_load(Group sg,
                  joint_matrix<S, Use, NumRows, NumCols, Layout, Group> &res,
                  multi_ptr<T, Space, IsDecorated> src, size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  sycl::ext::oneapi::detail::load_multiplicand_cuda<S, T, NumRows, NumCols, Use,
                                                    Layout, Space>(
      res.cuda_impl, src, stride);
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = res;
  std::ignore = src;
  std::ignore = stride;
  throw runtime_error(
      "This version of the matrix extension is only currently supported on "
      "Nvidia devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          access::address_space Space, access::decorated IsDecorated>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group sg,
    joint_matrix<T, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                 Group> &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  sycl::ext::oneapi::detail::joint_matrix_store_cuda<T, NumRows, NumCols,
                                                     Space>(src.cuda_impl, dst,
                                                            stride, Layout);
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = src;
  std::ignore = dst;
  std::ignore = stride;
  throw runtime_error(
      "This version of the matrix extension is only currently supported on "
      "Nvidia devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename Ta, typename Tb, typename Tc, std::size_t M,
          std::size_t K, std::size_t N, layout LayoutA, layout LayoutB>
inline __SYCL_ALWAYS_INLINE
    joint_matrix<Tc, use::accumulator, M, N,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                 Group>
    joint_matrix_mad(
        Group sg, joint_matrix<Ta, use::a, M, K, LayoutA, Group> &A,
        joint_matrix<Tb, use::b, K, N, LayoutB, Group> &B,
        joint_matrix<Tc, use::accumulator, M, N,
                     sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                     Group> &C) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  if constexpr (std::is_same<Ta, Tb>::value) {
    joint_matrix<Tc, use::accumulator, M, N,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic,
                 Group>
        D;
    sycl::ext::oneapi::detail::joint_matrix_mad_cuda<Ta, Tc, M, K, N, LayoutA,
                                                     LayoutB>(
        D.cuda_impl, A.cuda_impl, B.cuda_impl, C.cuda_impl);
    return D;
  } else {
    assert(false && "Ta != Tb : In the CUDA backend joint_matrix_mad "
                    "requires that joint_matrix data types Ta and Tb match");
  }
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = A;
  std::ignore = B;
  std::ignore = C;
  throw runtime_error(
      "This version of the matrix extension is only currently supported on "
      "Nvidia devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

// This function rounds the bottom 13 bits up or down, and then zeros out the
// bottom bits
inline __SYCL_ALWAYS_INLINE float round_to_tf32(float &a) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  int32_t tmp_int = __nvvm_f2tf32_rna(a);
  return __nvvm_bitcast_i2f(tmp_int);
#else
  uint32_t tmp_uint = reinterpret_cast<uint32_t &>(a);
  tmp_uint += 0x1000u;
  tmp_uint &= 0xFFFFE000u;
  float ret = reinterpret_cast<float &>(tmp_uint);
  return ret;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
}

} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
