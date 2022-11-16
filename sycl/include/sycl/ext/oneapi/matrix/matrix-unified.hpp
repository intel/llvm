//===------- matrix-unified.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include "matrix-intel.hpp"

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
// nv's impl
#else
  __spv::__spirv_JointMatrixINTEL<
      T, Rows, Cols, spv_matrix_layout_traits<Layout>::value,
      spv_scope_traits<Group>::value, spv_matrix_use_traits<Use>::value> *spvm;
#endif
#endif
  template <typename T2, size_t nRows, size_t nCols, use UseT, layout LayoutT,
            typename Grp>
  friend class wi_element;
  template <typename T2, size_t nRows, size_t nCols, use UseT, layout LayoutT,
            typename Grp>
  friend class wi_data;

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
  joint_matrix(Group sg) {
#ifndef __SYCL_DEVICE_ONLY__
    std::ignore = sg;
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif // __SYCL_DEVICE_ONLY__
  }
  inline __SYCL_ALWAYS_INLINE decltype(auto) get_wi_data() {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
    // return wi_data(cuda_impl.wi_marray);
#else
    return wi_data<T, Rows, Cols, Use, Layout, Group>(*this);
#endif
#else
    // Host version of get_wi_data required by compiler even though it will
    // never be called because joint_matrix cannot be constructed on host.
#if defined(__NVPTX__)
    // return wi_data(cuda_impl.wi_marray);
    if constexpr (std::is_same_v<T, precision::tf32>) {
      marray<float, 1> unused{};
      return wi_data<float, 1>(unused);
    } else {
      marray<T, 1> unused{};
      return wi_data<T, 1>(unused);
    }
#else
    return wi_data<T, Rows, Cols, Use, Layout, Group>(*this);
#endif
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
  // res.cuda_impl.wi_marray = v;
#else
  res.spvm =
      __spirv_CompositeConstruct<T, NumRows, NumCols,
                                 spv_matrix_use_traits<Use>::value,
                                 spv_matrix_layout_traits<Layout>::value>(
          static_cast<T>(v));
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
  // nv's impl
  // sycl::ext::oneapi::detail::load_accumulator_cuda(res.cuda_impl, src,
  // stride,
  //                                                  Layout);
#else
  // intel's impl
  // matL is determined by matrix.use?
  T *Ptr = src.get();
  switch (Layout) {
  default:
    assert(false && "Invalid Memory Layout!");
  case layout::row_major:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::col_major:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::packed:
    res.spvm = __spirv_JointMatrixLoadINTEL<
        T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
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
  // nv's impl
  // sycl::ext::oneapi::detail::load_multiplicand_cuda<S, T, NumRows, NumCols,
  // Use,
  //                                                   Layout, Space>(
  //     res.cuda_impl, src, stride);
#else
  T *Ptr = src.get();
  res.spvm =
      __spirv_JointMatrixLoadINTEL<T, NumRows, NumCols,
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
  // nv's impl
  // sycl::ext::oneapi::detail::joint_matrix_store_cuda<T, NumRows, NumCols,
  //                                                    Space>(src.cuda_impl,
  //                                                    dst,
  //                                                           stride, Layout);
#else
  // intel's impl
  T *Ptr = dst.get();
  switch (Layout) {
  default:
    assert(false && "Invalid Memory Layout!");
  case layout::row_major:
    __spirv_JointMatrixStoreINTEL<
        T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::RowMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::col_major:
    __spirv_JointMatrixStoreINTEL<
        T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
        spv_matrix_layout_traits<layout::dynamic>::value>(
        Ptr, src.spvm, stride, __spv::MatrixLayout::ColumnMajor,
        spv_scope_traits<Group>::value);
    break;
  case layout::packed:
    __spirv_JointMatrixStoreINTEL<
        T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
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
        D(sg);
    sycl::ext::oneapi::detail::joint_matrix_mad_cuda<Ta, Tc, M, K, N, LayoutA,
                                                     LayoutB>(
        D.cuda_impl, A.cuda_impl, B.cuda_impl, C.cuda_impl);
    return D;
  } else {
    assert(false && "Ta != Tb : In the CUDA backend joint_matrix_mad "
                    "requires that joint_matrix data types Ta and Tb match");
  }
#else
  joint_matrix<Tc, use::accumulator, M, N, layout::dynamic, Group> res(sg);
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
  throw runtime_error(
      "This version of the matrix extension is only currently supported on "
      "Nvidia devices",
      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
