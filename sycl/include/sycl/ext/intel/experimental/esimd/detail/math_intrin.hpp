//==------------ math_intrin.hpp - DPC++ Explicit SIMD API -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares experimental Explicit SIMD math intrinsics.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#include <cmath>
#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/detail/host_util.hpp>
#include <sycl/ext/intel/esimd/detail/math_intrin.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>

#define __ESIMD_raw_vec_t(T, SZ)                                               \
  sycl::ext::intel::esimd::detail::vector_type_t<                              \
      sycl::ext::intel::esimd::detail::__raw_t<T>, SZ>
#define __ESIMD_cpp_vec_t(T, SZ)                                               \
  sycl::ext::intel::esimd::detail::vector_type_t<                              \
      sycl::ext::intel::esimd::detail::__cpp_t<T>, SZ>

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ssshl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ssshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_rol(__ESIMD_raw_vec_t(T1, SZ) src0, __ESIMD_raw_vec_t(T1, SZ) src1);
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ror(__ESIMD_raw_vec_t(T1, SZ) src0, __ESIMD_raw_vec_t(T1, SZ) src1);

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umulh(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smulh(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_frc(__ESIMD_DNS::vector_type_t<float, SZ> src0);

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_lzd(__ESIMD_raw_vec_t(T, SZ) src0);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_bfrev(__ESIMD_raw_vec_t(T1, SZ) src0);

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_bfi(__ESIMD_raw_vec_t(T0, SZ) src0, __ESIMD_raw_vec_t(T0, SZ) src1,
                __ESIMD_raw_vec_t(T0, SZ) src2, __ESIMD_raw_vec_t(T0, SZ) src3);

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sbfe(__ESIMD_raw_vec_t(T0, SZ) src0, __ESIMD_raw_vec_t(T0, SZ) src1,
                 __ESIMD_raw_vec_t(T0, SZ) src2);

template <typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __esimd_dp4(__ESIMD_raw_vec_t(T, N) v1, __ESIMD_raw_vec_t(T, N) v2)
#ifdef __SYCL_DEVICE_ONLY__
        ;
#else
{
  if constexpr (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  __ESIMD_raw_vec_t(T, N) retv;
  for (auto i = 0; i != N; i += 4) {
    T dp = (v1[i] * v2[i]) + (v1[i + 1] * v2[i + 1]) + (v1[i + 2] * v2[i + 2]) +
           (v1[i + 3] * v2[i + 3]);
    retv[i] = dp;
    retv[i + 1] = dp;
    retv[i + 2] = dp;
    retv[i + 3] = dp;
  }
  return retv.data();
}
#endif // __SYCL_DEVICE_ONLY__

#ifndef __SYCL_DEVICE_ONLY__

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ssshl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = ret;
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = ret;
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = ret;
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = ret;
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ssshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = __ESIMD_EMU_DNS::satur<T0>::template saturate<T1>(ret, 1);
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = __ESIMD_EMU_DNS::satur<T0>::template saturate<T1>(ret, 1);
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = __ESIMD_EMU_DNS::satur<T0>::template saturate<T1>(ret, 1);
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T1>::type ret;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i] << src1[i];
    retv[i] = __ESIMD_EMU_DNS::satur<T0>::template saturate<T1>(ret, 1);
  }
  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_rol(__ESIMD_raw_vec_t(T1, SZ) src0,
                __ESIMD_raw_vec_t(T1, SZ) src1) {
  __ESIMD_UNSUPPORTED_ON_HOST;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ror(__ESIMD_raw_vec_t(T1, SZ) src0,
                __ESIMD_raw_vec_t(T1, SZ) src1) {
  __ESIMD_UNSUPPORTED_ON_HOST;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umulh(__ESIMD_raw_vec_t(T, SZ) src0,
                  __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    unsigned long long temp;
    SIMDCF_ELEMENT_SKIP(i);
    temp = (long long)src0[i] * (long long)src1[i];
    retv[i] = temp >> 32;
  }
  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smulh(__ESIMD_raw_vec_t(T, SZ) src0,
                  __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    long long temp;
    SIMDCF_ELEMENT_SKIP(i);
    temp = (long long)src0[i] * (long long)src1[i];
    retv[i] = temp >> 32;
  }
  return retv;
}

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_frc(__ESIMD_DNS::vector_type_t<float, SZ> src0) {
  __ESIMD_DNS::vector_type_t<float, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = src0[i] - floor(src0[i]);
  }
  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_lzd(__ESIMD_raw_vec_t(T, SZ) src0) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  T ret;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i];
    uint32_t cnt = 0;
    while ((ret & 1u << 31u) == 0 && cnt != 32) {
      cnt++;
      ret = ret << 1;
    }
    retv[i] = cnt;
  }

  return retv;
}

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_bfrev(__ESIMD_raw_vec_t(T1, SZ) src0) {
  int i, j;
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T1>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  __ESIMD_raw_vec_t(T0, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    T0 input = src0[i];
    T0 output = 0;
    for (j = 0; j < sizeof(T0) * 8; j++) {
      output |= input & 0x1;

      // Don't shift if this was the last one
      if ((j + 1) < (sizeof(T0) * 8)) {
        output <<= 1;
        input >>= 1;
      }
    }
    retv[i] = output;
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_bfi(__ESIMD_raw_vec_t(T, SZ) width, __ESIMD_raw_vec_t(T, SZ) offset,
                __ESIMD_raw_vec_t(T, SZ) val, __ESIMD_raw_vec_t(T, SZ) src) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T>::type ret;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    const uint32_t mask = ((1 << width[i]) - 1) << offset[i];
    const uint32_t imask = ~mask;
    ret = (src[i] & imask) | ((val[i] << offset[i] & mask));
    // Sign extend if signed type
    if constexpr (std::is_signed_v<T>) {
      int m = 1U << (width[i] - 1);
      ret = (ret ^ m) - m;
    }
    retv[i] = ret;
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_sbfe(__ESIMD_raw_vec_t(T, SZ) width,
                 __ESIMD_raw_vec_t(T, SZ) offset,
                 __ESIMD_raw_vec_t(T, SZ) src) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::maxtype<T>::type ret;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    const uint32_t mask = ((1 << width[i]) - 1) << offset[i];
    ret = (src[i] & mask) >> offset[i];
    retv[i] = ret;
  }

  return retv;
}

inline constexpr __ESIMD_NS::uint
__esimd_dpas_bits_precision(__ESIMD_XMX_NS::dpas_argument_type precisionType) {
  return precisionType == __ESIMD_XMX_NS::dpas_argument_type::tf32 ? 32
         : precisionType == __ESIMD_XMX_NS::dpas_argument_type::bf16 ||
                 precisionType == __ESIMD_XMX_NS::dpas_argument_type::fp16
             ? 16
         : precisionType == __ESIMD_XMX_NS::dpas_argument_type::s8 ||
                 precisionType == __ESIMD_XMX_NS::dpas_argument_type::u8
             ? 8
         : precisionType == __ESIMD_XMX_NS::dpas_argument_type::s4 ||
                 precisionType == __ESIMD_XMX_NS::dpas_argument_type::u4
             ? 4
         : precisionType == __ESIMD_XMX_NS::dpas_argument_type::s2 ||
                 precisionType == __ESIMD_XMX_NS::dpas_argument_type::u2
             ? 2
             : 1;
}

template <__ESIMD_XMX_NS::dpas_argument_type src1_precision,
          __ESIMD_XMX_NS::dpas_argument_type src2_precision, int systolic_depth,
          int repeat_count, typename RT, typename T0, typename T1, typename T2,
          __ESIMD_NS::uint SZ, __ESIMD_NS::uint N1, __ESIMD_NS::uint N2>
inline __ESIMD_DNS::vector_type_t<RT, SZ>
__esimd_dpas_inner(const __ESIMD_DNS::vector_type_t<T0, SZ> *src0,
                   const __ESIMD_DNS::vector_type_t<T1, N1> &src1,
                   const __ESIMD_DNS::vector_type_t<T2, N2> &src2) {
  __ESIMD_DNS::vector_type_t<RT, SZ> retv;

  __ESIMD_NS::uint sat1 =
      __ESIMD_EMU_DNS::SetSatur<
          T1, __ESIMD_EMU_DNS::is_inttype<RT>::value>::set() ||
      __ESIMD_EMU_DNS::SetSatur<T2,
                                __ESIMD_EMU_DNS::is_inttype<RT>::value>::set();

  __ESIMD_NS::uint V = 0, U = 0, k = 0, temp = 0, src1_ops_per_dword = 0, p = 0;

  constexpr auto src1_el_bits = __esimd_dpas_bits_precision(src1_precision);
  constexpr auto src2_el_bits = __esimd_dpas_bits_precision(src2_precision);

  constexpr auto max_el_bits = std::max(src1_el_bits, src2_el_bits);
  constexpr __ESIMD_NS::uint ops_per_chan =
      std::max(std::min(32 / max_el_bits, static_cast<uint32_t>(8)),
               static_cast<uint32_t>(1));

  uint32_t src1_signed =
      src1_precision == __ESIMD_XMX_NS::dpas_argument_type::s2 ||
              src1_precision == __ESIMD_XMX_NS::dpas_argument_type::s4 ||
              src1_precision == __ESIMD_XMX_NS::dpas_argument_type::s8
          ? 1
          : 0;

  uint32_t src2_signed =
      src2_precision == __ESIMD_XMX_NS::dpas_argument_type::s2 ||
              src2_precision == __ESIMD_XMX_NS::dpas_argument_type::s4 ||
              src2_precision == __ESIMD_XMX_NS::dpas_argument_type::s8
          ? 1
          : 0;

  constexpr uint32_t src1_vec_bit_size = sizeof(T1) * N1 * 8;
  constexpr uint32_t src1_num_elem = src1_vec_bit_size / src1_el_bits;
  constexpr size_t SIMDSize = src1_num_elem / (systolic_depth * ops_per_chan);
  static_assert(SIMDSize == 8 || SIMDSize == 16,
                "Execution size must be 8 or 16");
  constexpr bool isPvc = SIMDSize == 16;

  constexpr bool
      pvcHfDest = isPvc && std::is_same_v<RT, unsigned short> &&
                  src1_precision == __ESIMD_XMX_NS::dpas_argument_type::fp16 &&
                  src2_precision == __ESIMD_XMX_NS::dpas_argument_type::fp16,
      pvcHfSrc0 = isPvc && std::is_same_v<T0, unsigned short> &&
                  src1_precision == __ESIMD_XMX_NS::dpas_argument_type::fp16 &&
                  src2_precision == __ESIMD_XMX_NS::dpas_argument_type::fp16,
      pvcBfDest = isPvc && std::is_same_v<RT, unsigned short> &&
                  src1_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16 &&
                  src2_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16,
      pvcBfSrc0 = isPvc && std::is_same_v<T0, unsigned short> &&
                  src1_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16 &&
                  src2_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16,
      pvcBfOrHfDest = pvcBfDest || pvcHfDest,

      pvcBfDestChecks =
          pvcBfDest &&
          src1_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16 &&
          src2_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16,

      pvcHfDestChecks =
          pvcHfDest &&
          ((src1_precision == __ESIMD_XMX_NS::dpas_argument_type::fp16 &&
            src2_precision == __ESIMD_XMX_NS::dpas_argument_type::fp16) ||
           (src1_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16 &&
            src2_precision == __ESIMD_XMX_NS::dpas_argument_type::bf16)),

      destTypeChk =
          (!pvcBfOrHfDest && __ESIMD_EMU_DNS::is_fp_or_dword_type<RT>::value) ||
          (pvcBfOrHfDest && (pvcBfDestChecks || pvcHfDestChecks)),

      srcTypeChk = __ESIMD_EMU_DNS::is_dword_type<T1>::value &&
                   __ESIMD_EMU_DNS::is_dword_type<T2>::value,

      destSizeChk = SZ >= /*TODO: ==*/SIMDSize * repeat_count,

      systolicDepthAndRepeatCountChk =
          systolic_depth == 8 && repeat_count >= 1 && repeat_count <= 8,

      src1CountChk =
          N1 == ((src1_el_bits * systolic_depth * ops_per_chan * SZ) /
                 (repeat_count * sizeof(T1) * 8)),
      src2CountChk =
          N2 >= ((src2_el_bits * systolic_depth * ops_per_chan * repeat_count) /
                 (sizeof(T2) * 8))
      /*TODO: ==; fix PVCIGEMM24*/
      ;

  if constexpr (!isPvc)
    static_assert(!pvcBfOrHfDest, "dpas: hfloat and bfloat16 destination "
                                  "element type is only supported on PVC.");
  static_assert(destTypeChk, "dpas: unsupported dest and accumulator type.");
  static_assert(srcTypeChk, "dpas: unsupported src element type.");
  static_assert(destSizeChk,
                "dpas: destination size must be SIMDSize x repeat_count.");
  static_assert(systolicDepthAndRepeatCountChk,
                "dpas: only systolic_depth = 8 and repeat_count of 1 to 8 are "
                "supported.");
  static_assert(src1CountChk, "dpas: invalid size for src1.");
  static_assert(src2CountChk, "dpas: invalid size for src2.");

  using TmpAccEl = std::conditional_t<
      pvcBfOrHfDest, float,
      typename __ESIMD_EMU_DNS::restype_ex<
          RT, typename __ESIMD_EMU_DNS::restype_ex<T1, T2>::type>::type>;

  __ESIMD_DNS::vector_type_t<TmpAccEl, SIMDSize> simdAcc;

  for (unsigned r = 0; r < repeat_count; r++) {
    V = r;
    k = 0;

    for (unsigned n = 0; n < SIMDSize; n++) {
      if (src0 != nullptr) {
        auto src0El = src0[0][r * SIMDSize + n];

        if (pvcBfSrc0) {
          const auto tmp = (uint32_t)(src0El) << 16;
          simdAcc[n] = reinterpret_cast<const TmpAccEl &>(tmp);
        } else if (pvcHfSrc0) {
          simdAcc[n] = reinterpret_cast<const __ESIMD_EMU_DNS::half &>(src0El);
        } else
          simdAcc[n] = src0El;
      } else
        simdAcc[n] = 0;
    }

    for (unsigned s = 0; s < systolic_depth; s++) {
      src1_ops_per_dword = 32 / (ops_per_chan * src1_el_bits);
      // U = s / src1_ops_per_dword;
      U = s >> unsigned(std::log2(src1_ops_per_dword));

      for (unsigned n = 0; n < SIMDSize; n++) {
        for (unsigned d = 0; d < ops_per_chan; d++) {
          p = d + (s % src1_ops_per_dword) * ops_per_chan;
          uint32_t extension_temp = false;

          if constexpr (src2_precision ==
                        __ESIMD_XMX_NS::dpas_argument_type::bf16) {
            const auto s1 =
                extract<uint32_t>(src1_el_bits, p * src1_el_bits,
                                  src1[U * SIMDSize + n], extension_temp)
                << 16;
            const auto s2 =
                extract<uint32_t>(src2_el_bits, d * src2_el_bits,
                                  src2[V * 8 + k / ops_per_chan], src2_signed)
                << 16;
            simdAcc[n] += reinterpret_cast<const float &>(s2) *
                          reinterpret_cast<const float &>(s1);
          } else if constexpr (src2_precision ==
                               __ESIMD_XMX_NS::dpas_argument_type::fp16) {
            const auto s1 =
                extract<short>(src1_el_bits, p * src1_el_bits,
                               src1[U * SIMDSize + n], extension_temp);
            const auto s2 =
                extract<short>(src2_el_bits, d * src2_el_bits,
                               src2[V * 8 + k / ops_per_chan], src2_signed);
            simdAcc[n] += reinterpret_cast<const __ESIMD_EMU_DNS::half &>(s1) *
                          reinterpret_cast<const __ESIMD_EMU_DNS::half &>(s2);
          } else {
            int src = (sizeof(T2) * 8) / (ops_per_chan * src2_el_bits);
            int off = s % src * (ops_per_chan * src2_el_bits);
            int src1_tmp = extract<T1>(src1_el_bits, p * src1_el_bits,
                                       src1[U * SIMDSize + n], src1_signed);
            int src2_tmp = extract<T2>(src2_el_bits, d * src2_el_bits + off,
                                       src2[(V * 8 + k / ops_per_chan) / src],
                                       src2_signed);
            simdAcc[n] += src1_tmp * src2_tmp;
          }
        }
      }

      k += ops_per_chan;

    } // Systolic phase.

    for (unsigned n = 0; n < SIMDSize; n++) {
      if constexpr (pvcBfDest) {
        // TODO: make abstraction, support saturation, review rounding algo for
        // corner cases.
        auto tmpFloat = simdAcc[n];
        auto tmpUint = reinterpret_cast<uint32_t &>(tmpFloat);
        if (std::isnormal(tmpFloat) && tmpUint & 1ull << 15 &&
            (tmpUint & 0x7fff || tmpUint & 1ull << 16)) {
          tmpUint += 1ull << 16;
        }
        retv[r * SIMDSize + n] =
            static_cast<short>(reinterpret_cast<uint32_t &>(tmpUint) >> 16);
      } else if constexpr (pvcHfDest) {
        retv[r * SIMDSize + n] =
            __ESIMD_EMU_DNS::satur<sycl::half>::saturate<TmpAccEl>(simdAcc[n],
                                                                   sat1);
      } else
        retv[r * SIMDSize + n] =
            __ESIMD_EMU_DNS::satur<RT>::template saturate<TmpAccEl>(simdAcc[n],
                                                                    sat1);
    }

  } // Repeat.

  return retv;
}
#endif // #ifndef __SYCL_DEVICE_ONLY__

template <__ESIMD_XMX_NS::dpas_argument_type src1_precision,
          __ESIMD_XMX_NS::dpas_argument_type src2_precision, int systolic_depth,
          int repeat_count, typename T, typename T0, typename T1, typename T2,
          int N, int N1, int N2, int res_sign = std::is_signed_v<T>,
          int acc_sign = std::is_signed_v<T0>>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N>
__esimd_dpas2(__ESIMD_DNS::vector_type_t<T0, N> src0,
              __ESIMD_DNS::vector_type_t<T1, N1> src1,
              __ESIMD_DNS::vector_type_t<T2, N2> src2)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // !__SYCL_DEVICE_ONLY__
{
  return __esimd_dpas_inner<src1_precision, src2_precision, systolic_depth,
                            repeat_count, T, T0, T1, T2, N, N1, N2>(
      std::addressof(src0), src1, src2);
}
#endif // !__SYCL_DEVICE_ONLY__

template <int Info, typename T, typename T1, typename T2, int N, int N1, int N2>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N>
__esimd_dpas_nosrc0(__ESIMD_DNS::vector_type_t<T1, N1> src1,
                    __ESIMD_DNS::vector_type_t<T2, N2> src2)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // !__SYCL_DEVICE_ONLY__
{
  constexpr __ESIMD_XMX_NS::dpas_argument_type src1_precision =
      static_cast<__ESIMD_XMX_NS::dpas_argument_type>(Info & 0xff);
  constexpr __ESIMD_XMX_NS::dpas_argument_type src2_precision =
      static_cast<__ESIMD_XMX_NS::dpas_argument_type>((Info >> 8) & 0xff);
  constexpr int systolic_depth = (Info >> 16) & 0xff;
  constexpr int repeat_count = (Info >> 24) & 0xff;
  return __esimd_dpas_inner<src1_precision, src2_precision, systolic_depth,
                            repeat_count, T, T, T1, T2, N, N1, N2>(nullptr,
                                                                   src1, src2);
}
#endif // !__SYCL_DEVICE_ONLY__

template <int Info, typename T, typename T1, typename T2, int N, int N1, int N2>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N>
__esimd_dpasw(__ESIMD_DNS::vector_type_t<T, N> src0,
              __ESIMD_DNS::vector_type_t<T1, N1> src1,
              __ESIMD_DNS::vector_type_t<T2, N2> src2)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // !__SYCL_DEVICE_ONLY__
{
  __ESIMD_UNSUPPORTED_ON_HOST;
  return __ESIMD_DNS::vector_type_t<T, N>();
}
#endif // !__SYCL_DEVICE_ONLY__

template <int Info, typename T, typename T1, typename T2, int N, int N1, int N2>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N>
__esimd_dpasw_nosrc0(__ESIMD_DNS::vector_type_t<T1, N1> src1,
                     __ESIMD_DNS::vector_type_t<T2, N2> src2)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // !__SYCL_DEVICE_ONLY__
{
  __ESIMD_UNSUPPORTED_ON_HOST;
  return __ESIMD_DNS::vector_type_t<T, N>();
}
#endif // !__SYCL_DEVICE_ONLY__

template <typename T, int N>
__ESIMD_INTRIN std::pair<__ESIMD_DNS::vector_type_t<T, N>,
                         __ESIMD_DNS::vector_type_t<T, N>>
__esimd_addc(__ESIMD_DNS::vector_type_t<T, N> src0,
             __ESIMD_DNS::vector_type_t<T, N> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // !__SYCL_DEVICE_ONLY__
{
  __ESIMD_NS::simd<uint64_t, N> Result64 = __ESIMD_NS::simd<T, N>(src0);
  Result64 += __ESIMD_NS::simd<T, N>(src1);
  auto Result32 = Result64.template bit_cast_view<T>();
  __ESIMD_NS::simd<uint32_t, N> CarryV = Result32.template select<N, 2>(1);
  __ESIMD_NS::simd<uint32_t, N> ResV = Result32.template select<N, 2>(0);
  std::pair<__ESIMD_DNS::vector_type_t<T, N>, __ESIMD_DNS::vector_type_t<T, N>>
      ReturnValue = std::make_pair(CarryV.data(), ResV.data());
  return ReturnValue;
}
#endif // !__SYCL_DEVICE_ONLY__

template <typename T, int N>
__ESIMD_INTRIN std::pair<__ESIMD_DNS::vector_type_t<T, N>,
                         __ESIMD_DNS::vector_type_t<T, N>>
__esimd_subb(__ESIMD_DNS::vector_type_t<T, N> src0,
             __ESIMD_DNS::vector_type_t<T, N> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // !__SYCL_DEVICE_ONLY__
{
  __ESIMD_NS::simd<uint64_t, N> Result64 = __ESIMD_NS::simd<T, N>(src0);
  Result64 -= __ESIMD_NS::simd<T, N>(src1);
  auto Result32 = Result64.template bit_cast_view<T>();
  __ESIMD_NS::simd<uint32_t, N> BorrowV =
      __ESIMD_NS::simd<T, N>(src0) < __ESIMD_NS::simd<T, N>(src1);
  __ESIMD_NS::simd<uint32_t, N> ResV = Result32.template select<N, 2>(0);
  std::pair<__ESIMD_DNS::vector_type_t<T, N>, __ESIMD_DNS::vector_type_t<T, N>>
      ReturnValue = std::make_pair(BorrowV.data(), ResV.data());
  return ReturnValue;
}
#endif // !__SYCL_DEVICE_ONLY__

template <uint8_t FuncControl, typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __esimd_bfn(__ESIMD_raw_vec_t(T, N) src0, __ESIMD_raw_vec_t(T, N) src1,
                __ESIMD_raw_vec_t(T, N) src2)
#ifdef __SYCL_DEVICE_ONLY__
        ;
#else  // !__SYCL_DEVICE_ONLY__
{
  __ESIMD_UNSUPPORTED_ON_HOST;
  return __ESIMD_DNS::vector_type_t<T, N>();
}
#endif // !__SYCL_DEVICE_ONLY__

#undef __ESIMD_raw_vec_t
#undef __ESIMD_cpp_vec_t

/// @endcond ESIMD_DETAIL
