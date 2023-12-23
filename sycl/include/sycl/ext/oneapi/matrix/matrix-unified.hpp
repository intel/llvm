//===------- matrix-unified.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include "matrix-intel.hpp"

#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
#include "matrix-tensorcores.hpp"
#elif defined(__gfx90a__)
#include "matrix-hip.hpp"
#endif // defined(__NVPTX__)
#endif // defined(__SYCL_DEVICE_ONLY__)

#include <sycl/access/access.hpp>             // for address_space
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_...
#include <sycl/detail/pi.h>                   // for PI_ERROR_INVAL...
#include <sycl/exception.hpp>                 // for runtime_error
#include <sycl/ext/oneapi/matrix/matrix-unified-utils.hpp> // for layout, use, tf32, convertMatrixUseEnumToString
#include <sycl/ext/oneapi/matrix/query-types.hpp> // for convertTypeToMatrixTypeString
#include <sycl/marray.hpp>                        // for marray
#include <sycl/multi_ptr.hpp>                     // for multi_ptr

#include <cstring>     // for size_t, memcpy
#include <stdint.h>    // for uint32_t
#include <tuple>       // for ignore, _Swall...
#include <type_traits> // for is_same, remov...

namespace sycl {
inline namespace _V1 {
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
      matrix_impl;
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  sycl::ext::oneapi::detail::joint_matrix_hip<T, Use, Rows, Cols, Layout>
      matrix_impl;
#elif defined(__SPIR__)
  __spv::__spirv_JointMatrixINTEL<
      T, Rows, Cols, spv_matrix_layout_traits<Layout>::value,
      spv_scope_traits<Group>::value, spv_matrix_use_traits<Use>::value> *spvm;
#else
  static_assert(false, "The joint_matrix API is only supported by the Intel, "
                       "CUDA and HIP (GFX90A) backends");
#endif // defined(__NVPTX__)
#endif // defined(__SYCL_DEVICE_ONLY__)

#if defined(__SYCL_DEVICE_ONLY__)
  [[__sycl_detail__::add_ir_attributes_function(
      "sycl-joint-matrix-type", "sycl-joint-matrix-use",
      "sycl-joint-matrix-rows", "sycl-joint-matrix-cols",
      sycl::detail::convertTypeToMatrixTypeString<T>(),
      sycl::detail::convertMatrixUseEnumToString(Use), Rows, Cols)]]
#endif // defined(__SYCL_DEVICE_ONLY__)
  joint_matrix() {
#ifndef __SYCL_DEVICE_ONLY__
    throw runtime_error("joint matrix is not supported on host device.",
                        PI_ERROR_INVALID_DEVICE);
#endif
  }
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__SPIR__)
  joint_matrix(const joint_matrix &other) = delete;
  joint_matrix &operator=(const joint_matrix &rhs) = delete;
#endif // defined(__SPIR__)
#endif
};

template <typename Group, typename T, use Use, size_t M, size_t N,
          layout Layout, typename F>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_apply(Group sg, joint_matrix<Group, T, Use, M, N, Layout> &jm,
                   F &&lambda) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__) || defined(__HIP_PLATFORM_AMD_MFMA__)
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
    lambda(element);
    wi_data_c[i] = element;
  }
#endif
#else
  std::ignore = sg;
  std::ignore = jm;
  std::ignore = lambda;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
  return;
}

template <typename Group, typename T, size_t NumRows, size_t NumCols, use Use,
          layout Layout, typename T2>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_fill(Group,
                  joint_matrix<Group, T, Use, NumRows, NumCols, Layout> &res,
                  const T2 &v) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__) || defined(__HIP_PLATFORM_AMD_MFMA__)
  res.matrix_impl.wi_marray = v;
#else
  using storage_element_type =
      typename oneapi::detail::jm_type_interpretation_helper_trait<
          T>::storage_element_type;
  res.spvm =
      __spirv_CompositeConstruct<storage_element_type, T, NumRows, NumCols,
                                 spv_matrix_use_traits<Use>::value,
                                 spv_matrix_layout_traits<Layout>::value>(
          static_cast<storage_element_type>(v));
#endif // defined(__NVPTX__)
#else
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
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support load from private memory!");
#if defined(__NVPTX__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::load_accumulator_cuda(res.matrix_impl, src, stride,
                                                   Layout);
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  sycl::ext::oneapi::detail::load_accumulator_hip(res.matrix_impl, src, stride,
                                                  Layout, sg);
#else
  std::ignore = sg;
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(src);
  res.spvm = __spirv_JointMatrixLoadINTEL<
      DecorT, S, NumRows, NumCols,
      spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, stride, sycl::detail::joint_matrix_layout_to_spv(Layout),
      spv_scope_traits<Group>::value);
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
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support load from private memory!");
#if defined(__NVPTX__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::load_multiplicand_cuda<S, T, NumRows, NumCols, Use,
                                                    Layout, Space>(
      res.matrix_impl, src, stride);
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  sycl::ext::oneapi::detail::load_multiplicand_hip<Group, S, T, NumRows,
                                                   NumCols, Use, Layout, Space>(
      res.matrix_impl, src, stride, sg);
#else
  std::ignore = sg;
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(src);
  res.spvm =
      __spirv_JointMatrixLoadINTEL<DecorT, S, NumRows, NumCols,
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

template <typename Group, typename S, typename T, size_t NumRows,
          size_t NumCols, typename PropertyListT,
          std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value,
                           bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg,
    joint_matrix<Group, S, use::accumulator, NumRows, NumCols,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic> &res,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> src,
    size_t stride, sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  throw runtime_error("Use joint_matrix_load on multi_ptr on Nvidia device.",
                      PI_ERROR_INVALID_DEVICE);
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  throw runtime_error("Use joint_matrix_load on multi_ptr on AMD device.",
                      PI_ERROR_INVALID_DEVICE);
#else
  std::ignore = sg;
  T *Ptr = src.get();
  res.spvm = __spirv_JointMatrixLoadINTEL<
      T, S, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, stride, sycl::detail::joint_matrix_layout_to_spv(Layout),
      spv_scope_traits<Group>::value);
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
    size_t NumCols, matrix::layout Layout, typename PropertyListT,
    std::enable_if_t<std::is_same<S, std::remove_const_t<T>>::value ||
                         (std::is_same<S, precision::tf32>::value &&
                          std::is_same<std::remove_const_t<T>, float>::value),
                     bool> = true>
inline __SYCL_ALWAYS_INLINE void joint_matrix_load(
    Group sg, joint_matrix<Group, S, Use, NumRows, NumCols, Layout> &res,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> src,
    size_t stride) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  throw runtime_error("Use joint_matrix_load on multi_ptr on Nvidia device.",
                      PI_ERROR_INVALID_DEVICE);
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  throw runtime_error("Use joint_matrix_load on multi_ptr on AMD device.",
                      PI_ERROR_INVALID_DEVICE);
#else
  std::ignore = sg;
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
    const joint_matrix<Group, T, use::accumulator, NumRows, NumCols,
                       sycl::ext::oneapi::experimental::matrix::layout::dynamic>
        &src,
    multi_ptr<T, Space, IsDecorated> dst, size_t stride,
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
  static_assert(Space != access::address_space::private_space,
                "Joint Matrix doesn't support store to private memory!");
#if defined(__NVPTX__)
  std::ignore = sg;
  sycl::ext::oneapi::detail::joint_matrix_store_cuda<T, NumRows, NumCols,
                                                     Space>(
      src.matrix_impl, dst, stride, Layout);
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  sycl::ext::oneapi::detail::joint_matrix_store_hip<Group, T, NumRows, NumCols,
                                                    Space>(src.matrix_impl, dst,
                                                           stride, Layout, sg);
#else
  std::ignore = sg;
  using DecorT = typename sycl::detail::DecoratedType<T, Space>::type;
  DecorT *Ptr = sycl::detail::getDecorated<DecorT>(dst);
  __spirv_JointMatrixStoreINTEL<
      DecorT, T, NumRows, NumCols,
      spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, src.spvm, stride, sycl::detail::joint_matrix_layout_to_spv(Layout),
      spv_scope_traits<Group>::value);
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

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          typename PropertyListT>
inline __SYCL_ALWAYS_INLINE void joint_matrix_store(
    Group sg,
    const joint_matrix<Group, T, use::accumulator, NumRows, NumCols,
                       sycl::ext::oneapi::experimental::matrix::layout::dynamic>
        &src,
    ext::oneapi::experimental::annotated_ptr<T, PropertyListT> dst,
    size_t stride, sycl::ext::oneapi::experimental::matrix::layout Layout) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  throw runtime_error("Use joint_matrix_store on multi_ptr on Nvidia device.",
                      PI_ERROR_INVALID_DEVICE);
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  throw runtime_error("Use joint_matrix_store on multi_ptr on AMD device.",
                      PI_ERROR_INVALID_DEVICE);
#else
  std::ignore = sg;
  T *Ptr = dst.get();
  __spirv_JointMatrixStoreINTEL<
      T, T, NumRows, NumCols, spv_matrix_use_traits<use::accumulator>::value,
      spv_matrix_layout_traits<layout::dynamic>::value>(
      Ptr, src.spvm, stride, sycl::detail::joint_matrix_layout_to_spv(Layout),
      spv_scope_traits<Group>::value);
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

template <typename Group, typename Ta, typename Tb, typename Tc, typename Td,
          std::size_t M, std::size_t K, std::size_t N, layout LayoutA,
          layout LayoutB>
#if defined(__SYCL_DEVICE_ONLY__)
[[__sycl_detail__::add_ir_attributes_function(
    "sycl-joint-matrix-mad-type-A", "sycl-joint-matrix-mad-type-B",
    "sycl-joint-matrix-mad-type-C", "sycl-joint-matrix-mad-type-D",
    "sycl-joint-matrix-mad-size-M", "sycl-joint-matrix-mad-size-K",
    "sycl-joint-matrix-mad-size-N",
    sycl::detail::convertTypeToMatrixTypeString<Ta>(),
    sycl::detail::convertTypeToMatrixTypeString<Tb>(),
    sycl::detail::convertTypeToMatrixTypeString<Tc>(),
    sycl::detail::convertTypeToMatrixTypeString<Td>(), M, K, N)]]
#endif // defined(__SYCL_DEVICE_ONLY__)
inline __SYCL_ALWAYS_INLINE void
joint_matrix_mad(
    Group,
    joint_matrix<Group, Td, use::accumulator, M, N,
                 sycl::ext::oneapi::experimental::matrix::layout::dynamic> &D,
    const joint_matrix<Group, Ta, use::a, M, K, LayoutA> &A,
    const joint_matrix<Group, Tb, use::b, K, N, LayoutB> &B,
    const joint_matrix<Group, Tc, use::accumulator, M, N,
                       sycl::ext::oneapi::experimental::matrix::layout::dynamic>
        &C) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  if constexpr (std::is_same<Ta, Tb>::value) {
    sycl::ext::oneapi::detail::joint_matrix_mad_cuda<Ta, Tc, Td, M, K, N,
                                                     LayoutA, LayoutB>(
        D.matrix_impl, A.matrix_impl, B.matrix_impl, C.matrix_impl);
  } else {
    assert(false && "Ta != Tb : In the CUDA backend joint_matrix_mad "
                    "requires that joint_matrix data types Ta and Tb match");
  }
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  if constexpr (std::is_same<Ta, Tb>::value) {
    sycl::ext::oneapi::detail::joint_matrix_mad_hip<Ta, Tc, M, K, N, LayoutA,
                                                    LayoutB>(
        D.matrix_impl, A.matrix_impl, B.matrix_impl, C.matrix_impl);
  } else {
    assert(false && "Ta != Tb : In the HIP backend joint_matrix_mad "
                    "requires that joint_matrix data types Ta and Tb match");
  }
#else
  if constexpr (std::is_same<Ta, uint16_t>::value &&
                std::is_same<Tb, uint16_t>::value &&
                std::is_same<Tc, float>::value)
    D.spvm = __spirv_JointMatrixMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_unsigned<Ta>::value && std::is_unsigned<Tb>::value)
    D.spvm = __spirv_JointMatrixUUMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_signed<Ta>::value && std::is_unsigned<Tb>::value)
    D.spvm = __spirv_JointMatrixSUMadINTEL(A.spvm, B.spvm, C.spvm);
  else if constexpr (std::is_unsigned<Ta>::value && std::is_signed<Tb>::value)
    D.spvm = __spirv_JointMatrixUSMadINTEL(A.spvm, B.spvm, C.spvm);
  else
    D.spvm = __spirv_JointMatrixMadINTEL(A.spvm, B.spvm, C.spvm);
#endif // defined(__NVPTX__)
#else
  std::ignore = A;
  std::ignore = B;
  std::ignore = C;
  std::ignore = D;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <typename Group, typename T1, typename T2, size_t Rows, size_t Cols,
          use Use1, use Use2, layout Layout1, layout Layout2>
void joint_matrix_copy(
    Group sg, joint_matrix<Group, T1, Use1, Rows, Cols, Layout1> &src,
    joint_matrix<Group, T2, Use2, Rows, Cols, Layout2> &dst) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__) || defined(__HIP_PLATFORM_AMD_MFMA__)
  std::ignore = sg;
  dst.matrix_impl.wi_marray = src.matrix_impl.wi_marray;
#else
  using storage_element_type =
      typename oneapi::detail::jm_type_interpretation_helper_trait<
          T2>::storage_element_type;
  auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(sg, src);
  auto wi_data_dst = sycl::ext::oneapi::detail::get_wi_data(sg, dst);
  for (int i = 0; i < wi_data_c.length(); i++) {
    wi_data_dst[i] = static_cast<storage_element_type>(wi_data_c[i]);
  }
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = dst;
  std::ignore = src;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

// This function rounds the bottom 13 bits up or down, and then zeros out the
// bottom bits
inline __SYCL_ALWAYS_INLINE float round_to_tf32(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  int32_t tmp_int = __nvvm_f2tf32_rna(a);
  return __nvvm_bitcast_i2f(tmp_int);
#else
  return __spirv_RoundFToTF32INTEL(a);
#endif // defined(__NVPTX__)
#else
  uint32_t tmp_uint = reinterpret_cast<const uint32_t &>(a);
  tmp_uint += 0x1000u;
  tmp_uint &= 0xFFFFE000u;
  float ret = 0;
  std::memcpy(&ret, &tmp_uint, sizeof(float));
  return ret;
#endif // defined(__SYCL_DEVICE_ONLY__)
}

template <size_t NumRows, size_t NumCols, typename Group, typename T,
          typename Properties = ext::oneapi::experimental::empty_properties_t>
inline __SYCL_ALWAYS_INLINE void
joint_matrix_prefetch(Group sg, T *Ptr, size_t stride,
                      sycl::ext::oneapi::experimental::matrix::layout Layout,
                      Properties properties = {}) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  std::ignore = sg;
  std::ignore = properties;
  throw runtime_error(
      "joint_matrix_prefetch is not supported on Nvidia device.",
      PI_ERROR_INVALID_DEVICE);
#elif defined(__HIP_PLATFORM_AMD_MFMA__)
  std::ignore = sg;
  std::ignore = properties;
  throw runtime_error("joint_matrix_prefetch is not supported on AMD device.",
                      PI_ERROR_INVALID_DEVICE);
#else
  std::ignore = sg;
  auto prop = properties.template get_property<prefetch_hint_key>();
  // Will be removed once SPIRV implementation also uses offsetpointer
  size_t coordX = 0;
  size_t coordY = 0;
  __spirv_JointMatrixPrefetchINTEL<T, NumRows, NumCols>(
      Ptr, coordX, coordY, detail::PropertyMetaInfo<decltype(prop)>::value,
      sycl::detail::joint_matrix_layout_to_spv(Layout), stride);
#endif // defined(__NVPTX__)
#else
  std::ignore = sg;
  std::ignore = Ptr;
  std::ignore = stride;
  std::ignore = Layout;
  std::ignore = properties;
  throw runtime_error("joint matrix is not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif // defined(__SYCL_DEVICE_ONLY__)
}

} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
