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
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ssshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_rol(__ESIMD_raw_vec_t(T1, SZ) src0,
                __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ror(__ESIMD_raw_vec_t(T1, SZ) src0,
                __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umulh(__ESIMD_raw_vec_t(T, SZ) src0,
                  __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smulh(__ESIMD_raw_vec_t(T, SZ) src0,
                  __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_frc(__ESIMD_DNS::vector_type_t<float, SZ> src0) __ESIMD_INTRIN_END;

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_lzd(__ESIMD_raw_vec_t(T, SZ) src0) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_bfrev(__ESIMD_raw_vec_t(T1, SZ) src0) __ESIMD_INTRIN_END;

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_bfi(__ESIMD_raw_vec_t(T0, SZ) src0, __ESIMD_raw_vec_t(T0, SZ) src1,
                __ESIMD_raw_vec_t(T0, SZ) src2,
                __ESIMD_raw_vec_t(T0, SZ) src3) __ESIMD_INTRIN_END;

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sbfe(__ESIMD_raw_vec_t(T0, SZ) src0, __ESIMD_raw_vec_t(T0, SZ) src1,
                 __ESIMD_raw_vec_t(T0, SZ) src2) __ESIMD_INTRIN_END;

template <typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __esimd_dp4(__ESIMD_raw_vec_t(T, N) v1,
                __ESIMD_raw_vec_t(T, N) v2) __ESIMD_INTRIN_END;

template <__ESIMD_XMX_NS::dpas_argument_type src1_precision,
          __ESIMD_XMX_NS::dpas_argument_type src2_precision, int systolic_depth,
          int repeat_count, typename T, typename T0, typename T1, typename T2,
          int N, int N1, int N2, int res_sign = std::is_signed_v<T>,
          int acc_sign = std::is_signed_v<T0>>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N>
__esimd_dpas2(__ESIMD_DNS::vector_type_t<T0, N> src0,
              __ESIMD_DNS::vector_type_t<T1, N1> src1,
              __ESIMD_DNS::vector_type_t<T2, N2> src2) __ESIMD_INTRIN_END;

template <int Info, typename T, typename T1, typename T2, int N, int N1, int N2>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N>
__esimd_dpas_nosrc0(__ESIMD_DNS::vector_type_t<T1, N1> src1,
                    __ESIMD_DNS::vector_type_t<T2, N2> src2) __ESIMD_INTRIN_END;

template <int Info, typename T, typename T1, typename T2, int N, int N1, int N2>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N>
__esimd_dpasw(__ESIMD_DNS::vector_type_t<T, N> src0,
              __ESIMD_DNS::vector_type_t<T1, N1> src1,
              __ESIMD_DNS::vector_type_t<T2, N2> src2) __ESIMD_INTRIN_END;

template <int Info, typename T, typename T1, typename T2, int N, int N1, int N2>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N> __esimd_dpasw_nosrc0(
    __ESIMD_DNS::vector_type_t<T1, N1> src1,
    __ESIMD_DNS::vector_type_t<T2, N2> src2) __ESIMD_INTRIN_END;

template <typename T, int N>
__ESIMD_INTRIN std::pair<__ESIMD_DNS::vector_type_t<T, N>,
                         __ESIMD_DNS::vector_type_t<T, N>>
__esimd_addc(__ESIMD_DNS::vector_type_t<T, N> src0,
             __ESIMD_DNS::vector_type_t<T, N> src1) __ESIMD_INTRIN_END;

template <typename T, int N>
__ESIMD_INTRIN std::pair<__ESIMD_DNS::vector_type_t<T, N>,
                         __ESIMD_DNS::vector_type_t<T, N>>
__esimd_subb(__ESIMD_DNS::vector_type_t<T, N> src0,
             __ESIMD_DNS::vector_type_t<T, N> src1) __ESIMD_INTRIN_END;

template <uint8_t FuncControl, typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __esimd_bfn(__ESIMD_raw_vec_t(T, N) src0, __ESIMD_raw_vec_t(T, N) src1,
                __ESIMD_raw_vec_t(T, N) src2) __ESIMD_INTRIN_END;

template <int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(sycl::half, N)
    __esimd_srnd(__ESIMD_DNS::vector_type_t<float, N> src1,
                 __ESIMD_DNS::vector_type_t<uint16_t, N> src2)
        __ESIMD_INTRIN_END;

__ESIMD_INTRIN __ESIMD_raw_vec_t(uint32_t, 4)
    __esimd_timestamp() __ESIMD_INTRIN_END;

#undef __ESIMD_raw_vec_t
#undef __ESIMD_cpp_vec_t

/// @endcond ESIMD_DETAIL
