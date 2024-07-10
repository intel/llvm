//==----------------- xmx/dpas.hpp - DPC++ Explicit SIMD API ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Explicit SIMD API for DPAS Intel(R) Xe Matrix eXtensions (Intel(R) XMX).
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/xmx/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/math_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/tfloat32.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace sycl {
inline namespace _V1 {

namespace ext::intel::esimd::xmx {

namespace detail {

template <typename T> constexpr dpas_argument_type dpas_precision_from_type() {
  if constexpr (std::is_same_v<T,
                               sycl::ext::intel::experimental::esimd::tfloat32>)
    return dpas_argument_type::tf32;
  else if constexpr (std::is_same_v<T, sycl::half>)
    return dpas_argument_type::fp16;
  else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>)
    return dpas_argument_type::bf16;
  else if constexpr (std::is_same_v<T, unsigned char>)
    return dpas_argument_type::u8;
  else if constexpr (__ESIMD_DNS::is_type<T, char, signed char>())
    return dpas_argument_type::s8;
  else
    return dpas_argument_type::Invalid;
}

template <dpas_argument_type T> constexpr int dpas_bitsize_from_precision() {
  if constexpr (T == dpas_argument_type::u2 || T == dpas_argument_type::s2)
    return 2;
  else if constexpr (T == dpas_argument_type::u4 || T == dpas_argument_type::s4)
    return 4;
  else if constexpr (T == dpas_argument_type::u8 || T == dpas_argument_type::s8)
    return 8;
  else if constexpr (T == dpas_argument_type::bf16 ||
                     T == dpas_argument_type::fp16)
    return 16;
  else if constexpr (T == dpas_argument_type::tf32)
    return 32;
  else
    return -1;
}

template <int RepeatCount, int AElemBitSize, int BElemBitSize, bool IsDPASW>
constexpr void verify_repeat_count() {
  static_assert(RepeatCount >= 1 && RepeatCount <= 8,
                "Repeat count must be within 1 to 8 range");

  if constexpr (IsDPASW && RepeatCount != 8) {
    static_assert(!(AElemBitSize == 2 && BElemBitSize > 4),
                  "Unsupported repeat count for DPASW operation");

    static_assert(
        RepeatCount == 4 ||
            (AElemBitSize != 2 && (AElemBitSize != 4 || BElemBitSize <= 4)),
        "Unsupported repeat count for DPASW operation");
  }
}

template <int SystolicDepth, int RepeatCount, typename T, typename CT,
          typename BT, typename AT, dpas_argument_type BPrecision,
          dpas_argument_type APrecision, int BN, int AN, bool IsDPASW = false>
constexpr int verify_parameters_and_deduce_exec_size() {

  static_assert(SystolicDepth == 8, "Systolic depth must be equal to 8");
  static_assert(
      APrecision != dpas_argument_type::Invalid &&
          BPrecision != dpas_argument_type::Invalid,
      "The types of dpas arguments are either incorrect or cannot be deduced."
      "Fix the types and/or explicitly specify them.");

  constexpr int AElemBitSize = dpas_bitsize_from_precision<APrecision>();
  constexpr int BElemBitSize = dpas_bitsize_from_precision<BPrecision>();
  static_assert(AElemBitSize != -1 && BElemBitSize != -1,
                "Cannot deduce element size of input arguments");
  verify_repeat_count<RepeatCount, AElemBitSize, BElemBitSize, IsDPASW>();

  constexpr int MaxElemBitSize =
      AElemBitSize > BElemBitSize ? AElemBitSize : BElemBitSize;
  constexpr int MaxElemsInDword = 32 / MaxElemBitSize;
  constexpr int OpsPerChannel =
      MaxElemsInDword > 8 ? 8 : (MaxElemsInDword < 1 ? 1 : MaxElemsInDword);

  // A(_Mx_K) * B(_Kx_N) + C(_Mx_N)
  // where:
  //   _M = RepeatCount;
  //   _K = SystolicDepth * OpsPerChannel;
  //   _N = ExecutionSize (unknown, but deducible), must be 8 or 16.
  constexpr int _M = RepeatCount;
  constexpr int _K = SystolicDepth * OpsPerChannel;

  // Compute _N (aka ExecutionSize) from the matrix B.
  // It has _K*_N elements of BPrecision type, and BN elements of BT type
  // hold those _K*_N*BPrecision bits, which let's us compute _N.
  constexpr int BMatrixBitSize = sizeof(BT) * BN * 8;
  constexpr int BNumElems = BMatrixBitSize / BElemBitSize;
  constexpr int _N = BNumElems / _K;
  static_assert(_K * _N == BNumElems, "Cannot deduce the execution size.");

  // Now verify that AN elements of AT type hold exactly _M*_K elements
  // of APrecision type/size. Similarly for B: BN elements of BT type must
  // hold _K*_N elements of BPrecision type/size.
  // DPASW accepts 2x less expected AN elements than regular DPAS.
  constexpr int AFactorForDPASW = IsDPASW ? 2 : 1;
  static_assert(_M * _K * AElemBitSize == AN * sizeof(AT) * 8 * AFactorForDPASW,
                "The first matrix multiplier has wrong size.");
  static_assert(_K * _N * BElemBitSize == BN * sizeof(BT) * 8,
                "The second matrix multiplier has wrong size.");

  // Execution size may be 8 or 16 depending on the target device.
  // User must check if used execution size is supported before calling DPAS.
  constexpr int ExecutionSize = _N;

  static_assert(ExecutionSize == 8 || (!IsDPASW && ExecutionSize == 16),
                "Execution size must be 8 or 16 for DPAS and 8 for DPASW.");

  if constexpr (APrecision == dpas_argument_type::fp16 ||
                BPrecision == dpas_argument_type::fp16) {
    if constexpr (ExecutionSize == 8) {
      static_assert(APrecision == BPrecision &&
                        __ESIMD_DNS::is_type<T, float>() &&
                        __ESIMD_DNS::is_type<CT, float>(),
                    "Unsupported DPAS types! The supported types are:\n"
                    " Result |   C   |   B  |  A  \n"
                    "   f    |   f   |  hf  |  hf \n");
    } else {
      static_assert(APrecision == BPrecision &&
                        __ESIMD_DNS::is_type<T, float, sycl::half>() &&
                        __ESIMD_DNS::is_type<CT, float, sycl::half>(),
                    "Unsupported DPAS types! The supported types are:\n"
                    " Result |   C   |   B  |  A  \n"
                    " f, hf  | f, hf |  hf  |  hf \n");
    }
  } else if constexpr (APrecision == dpas_argument_type::bf16 ||
                       BPrecision == dpas_argument_type::bf16) {
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    if constexpr (ExecutionSize == 8) {
      static_assert(APrecision == BPrecision &&
                        __ESIMD_DNS::is_type<T, float, bfloat16>() &&
                        __ESIMD_DNS::is_type<CT, float, bfloat16>(),
                    "Unsupported DPAS types! The supported types are:\n"
                    " Result |   C   |   B  |  A        \n"
                    "   f    |   f   |  bf  |  bf       \n");
    } else {
      static_assert(APrecision == BPrecision &&
                        __ESIMD_DNS::is_type<T, float, bfloat16>() &&
                        __ESIMD_DNS::is_type<CT, float, bfloat16>(),
                    "Unsupported DPAS types! The supported types are:\n"
                    " Result |   C   |   B  |  A        \n"
                    " f, bf  | f, bf |  bf  |  bf       \n");
    }
  } else if constexpr (APrecision == dpas_argument_type::tf32 ||
                       BPrecision == dpas_argument_type::tf32) {
    static_assert(ExecutionSize == 16,
                  "tf32 type can be used only with ExecutionSize=16");
    static_assert(APrecision == BPrecision && std::is_same_v<T, float> &&
                      std::is_same_v<CT, float>,
                  "Unsupported DPAS types! The supported types are:\n"
                  " Result |   C   |   B  |  A   \n"
                  "   f    |   f   | tf32 | tf32 \n");
  } else {
    static_assert((APrecision == dpas_argument_type::u2 ||
                   APrecision == dpas_argument_type::s2 ||
                   APrecision == dpas_argument_type::u4 ||
                   APrecision == dpas_argument_type::s4 ||
                   APrecision == dpas_argument_type::u8 ||
                   APrecision == dpas_argument_type::s8) &&
                      (BPrecision == dpas_argument_type::u2 ||
                       BPrecision == dpas_argument_type::s2 ||
                       BPrecision == dpas_argument_type::u4 ||
                       BPrecision == dpas_argument_type::s4 ||
                       BPrecision == dpas_argument_type::u8 ||
                       BPrecision == dpas_argument_type::s8),
                  "Unsupported DPAS types! The supported types are:\n"
                  " Result |   C   |        B         |           A      \n"
                  " ud, d  | ud, d | ub,b,u4,s4,u2,s2 | ub,b,u4,s4,u2,s2 \n");
  }
  return ExecutionSize;
}

} // namespace detail

/// @defgroup sycl_esimd_xmx_systolic_array_api Systolic Array APIs.
/// APIs below are used to implement dot product accumulate systolic functions
/// @ingroup sycl_esimd

/// @addtogroup sycl_esimd_xmx_systolic_array_api
/// @{
/// DPAS (Dot Product Accumulate Systolic)
/// Computes the result of matrix operations: Result = C + A x B;
/// @param C represents DPAS accumulator operand.
/// @param B represents the 2nd matrix multiplier. It must have the VNNI encoded
/// layout.
/// @param A represents the 1st matrix multiplier.
/// @return the vector value of DPAS computation result.
template <
    int SystolicDepth, int RepeatCount, typename T, typename CT, typename BT,
    typename AT,
    dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
    dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(),
    int N, int BN, int AN>
__ESIMD_NS::simd<T, N> dpas(__ESIMD_NS::simd<CT, N> C,
                            __ESIMD_NS::simd<BT, BN> B,
                            __ESIMD_NS::simd<AT, AN> A) {
  (void)detail::verify_parameters_and_deduce_exec_size<
      SystolicDepth, RepeatCount, T, CT, BT, AT, BPrecision, APrecision, BN,
      AN>();

  using MsgT = int;
  constexpr int ANCasted = AN * sizeof(AT) / sizeof(MsgT);
  constexpr int BNCasted = BN * sizeof(BT) / sizeof(MsgT);
  __ESIMD_NS::simd<MsgT, ANCasted> ACasted = A.template bit_cast_view<MsgT>();
  __ESIMD_NS::simd<MsgT, BNCasted> BCasted = B.template bit_cast_view<MsgT>();
  using CRawT = typename __ESIMD_NS::simd<CT, N>::raw_element_type;
  using RawT = typename __ESIMD_NS::simd<T, N>::raw_element_type;
  return __esimd_dpas2<BPrecision, APrecision, SystolicDepth, RepeatCount, RawT,
                       CRawT, MsgT, MsgT, N, BNCasted, ANCasted>(
      C.data(), BCasted.data(), ACasted.data());
}

/// DPAS (Dot Product Accumulate Systolic)
/// Computes the result of matrix operations: Result = A x B;
/// @param B represents the 2nd matrix multiplier. It must have the VNNI encoded
/// layout.
/// @param A represents the 1st matrix multiplier.
/// @return the vector value of DPAS computation result.
template <
    int SystolicDepth, int RepeatCount, typename T, typename BT, typename AT,
    dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
    dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(),
    int BN, int AN>
auto dpas(__ESIMD_NS::simd<BT, BN> B, __ESIMD_NS::simd<AT, AN> A) {

  constexpr int ExecutionSize =
      detail::verify_parameters_and_deduce_exec_size<SystolicDepth, RepeatCount,
                                                     T, T, BT, AT, BPrecision,
                                                     APrecision, BN, AN>();
  // Result(_Mx_N) = A(_Mx_K) * B(_Kx_N)
  // where:
  //   _M = RepeatCount;
  //   _K = SystolicDepth * OpsPerChannel;
  //   _N = ExecutionSize (unknown, but deducible), must be 8 or 16.
  constexpr int ResultN = RepeatCount * ExecutionSize;

  using MsgT = int;
  constexpr int ANCasted = AN * sizeof(AT) / sizeof(MsgT);
  constexpr int BNCasted = BN * sizeof(BT) / sizeof(MsgT);
  __ESIMD_NS::simd<MsgT, ANCasted> ACasted = A.template bit_cast_view<MsgT>();
  __ESIMD_NS::simd<MsgT, BNCasted> BCasted = B.template bit_cast_view<MsgT>();

  constexpr int Info = (RepeatCount << 24) + (SystolicDepth << 16) +
                       ((int)APrecision << 8) + (int)BPrecision;
  using RawT = typename __ESIMD_NS::simd<T, ResultN>::raw_element_type;
  __ESIMD_NS::simd<T, ResultN> Result =
      __esimd_dpas_nosrc0<Info, RawT, MsgT, MsgT, ResultN, BNCasted, ANCasted>(
          BCasted.data(), ACasted.data());
  return Result;
}

/// DPAS (Dot Product Accumulate Systolic)
/// Computes the result of matrix operations: Result = C + A x B;
/// @param C represents DPAS accumulator operand.
/// @param B represents the 2nd matrix multiplier. It must have the VNNI encoded
/// layout.
/// @param A represents the 1st matrix multiplier.
/// @return the vector value of DPAS computation result.
template <
    int SystolicDepth, int RepeatCount, typename T, typename BT, typename AT,
    dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
    dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(),
    int N, int BN, int AN>
__ESIMD_NS::simd<T, N> dpasw(__ESIMD_NS::simd<T, N> C,
                             __ESIMD_NS::simd<BT, BN> B,
                             __ESIMD_NS::simd<AT, AN> A) {

  constexpr bool IsDPASW = true;
  (void)detail::verify_parameters_and_deduce_exec_size<
      SystolicDepth, RepeatCount, T, T, BT, AT, BPrecision, APrecision, BN, AN,
      IsDPASW>();

  constexpr int ANCasted = AN * sizeof(AT) / sizeof(int);
  constexpr int BNCasted = BN * sizeof(BT) / sizeof(int);
  __ESIMD_NS::simd<int, ANCasted> ACasted = A.template bit_cast_view<int>();
  __ESIMD_NS::simd<int, BNCasted> BCasted = B.template bit_cast_view<int>();

  using RawT = typename __ESIMD_NS::simd<T, N>::raw_element_type;
  constexpr int Info = (RepeatCount << 24) + (SystolicDepth << 16) +
                       ((int)APrecision << 8) + (int)BPrecision;
  return __esimd_dpasw<Info, RawT, int, int, N, BNCasted, ANCasted>(
      C.data(), BCasted.data(), ACasted.data());
}

/// DPAS (Dot Product Accumulate Systolic)
/// Computes the result of matrix operations: Result = A x B;
/// @param B represents the 2nd matrix multiplier. It must have the VNNI encoded
/// layout.
/// @param A represents the 1st matrix multiplier.
/// @return the vector value of DPAS computation result.
template <
    int SystolicDepth, int RepeatCount, typename T, typename BT, typename AT,
    dpas_argument_type BPrecision = detail::dpas_precision_from_type<BT>(),
    dpas_argument_type APrecision = detail::dpas_precision_from_type<AT>(),
    int BN, int AN>
auto dpasw(__ESIMD_NS::simd<BT, BN> B, __ESIMD_NS::simd<AT, AN> A) {

  constexpr bool IsDPASW = true;
  constexpr int ExecutionSize = detail::verify_parameters_and_deduce_exec_size<
      SystolicDepth, RepeatCount, T, T, BT, AT, BPrecision, APrecision, BN, AN,
      IsDPASW>();

  // Result(_Mx_N) = A(_Mx_K) * B(_Kx_N)
  // where:
  //   _M = RepeatCount;
  //   _K = SystolicDepth * OpsPerChannel;
  //   _N = ExecutionSize (unknown, but deducible), must be 8 or 16.
  constexpr int ResultN = RepeatCount * ExecutionSize;

  constexpr int ANCasted = AN * sizeof(AT) / sizeof(int);
  constexpr int BNCasted = BN * sizeof(BT) / sizeof(int);
  __ESIMD_NS::simd<int, ANCasted> ACasted = A.template bit_cast_view<int>();
  __ESIMD_NS::simd<int, BNCasted> BCasted = B.template bit_cast_view<int>();

  using RawT = typename __ESIMD_NS::simd<T, ResultN>::raw_element_type;
  constexpr int Info = (RepeatCount << 24) + (SystolicDepth << 16) +
                       ((int)APrecision << 8) + (int)BPrecision;
  __ESIMD_NS::simd<T, ResultN> Result =
      __esimd_dpasw_nosrc0<Info, RawT, int, int, ResultN, BNCasted, ANCasted>(
          BCasted.data(), ACasted.data());
  return Result;
}

/// @} sycl_esimd_xmx_systolic_array_api

} // namespace ext::intel::esimd::xmx
} // namespace _V1
} // namespace sycl
