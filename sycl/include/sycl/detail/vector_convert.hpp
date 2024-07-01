//==-- vector_convert.hpp --- vec::convert implementation ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
/// Implementation of \c vec::convert
///
/// Implementation consists of a bunch of helper functions to do different
/// conversions, as well as a single entry point function, which performs a
/// dispatch into the right helper.
///
/// Helpers are made to match corresponding SPIR-V instructions defined in
/// section 3.42.11. Conversion Instructions (SPIR-V specification version 1.6,
/// revision 2, unified):
/// - \c OpConvertFToU - float to unsigned integer
/// - \c OpConvertFToS - float to signed integer
/// - \c OpConvertSToF - signed integer to float
/// - \c OpConvertUToF - unsigned integer to float
/// - \c OpUConvert - unsigned integer to unsigned integer
/// - \c OpSConvert - signed integer to signed integer
/// - \c OpConvertF - float to float
/// - \c OpSatConvertSToU - signed integer to unsigned integer
/// - \c OpSatConvertUToS - unsigned integer to signed integer
///
/// To get the right SPIR-V instruction emitted from SYCL code, we need to make
/// a call to a specific built-in in the following format:
///   \c __spirv_[Op]_R[DestType][N?][_RoundingMode?]
/// where:
/// - \c [Op] is the name of instruction from the list above, without "Op"
///   prefix.
/// - \c [DestType] is the name of scalar return type
/// - \c [N?] is vector size; omitted for scalars
/// - \c [RoundingMode?] is rounding mode suffix, can be omitted
///
/// Implementation below is essentially split into two parts: for host and for
/// device.
///
/// Host part is really simple, as we only have scalar conversions available in
/// there. Most of them are implemented as regular \c static_cast with exception
/// for float to integer conversions, which require rounding mode handling.
///
/// Device part is more complicated, because we need to generate calls to those
/// \c __spirv* built-ins. To do so, macro code generation is used: we emit
/// number of function templates, which are conditionally enabled using SFINAE
/// for each destination type and vector size to each call their own \c __spriv*
/// built-in.
///
/// Finally, there is single entry point which performs a dispatch to the right
/// helper depending on source and destination types.

#pragma once

#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/exception.hpp>                  // for errc

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16

#ifndef __SYCL_DEVICE_ONLY__
#include <cfenv> // for fesetround, fegetround
#endif

#include <type_traits>

// Enable on only intel devices.
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
extern "C" {
// For converting BF16 to other types.
extern __DPCPP_SYCL_EXTERNAL float __imf_bfloat162float(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat16_as_short(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat16_as_ushort(uint16_t x);

// For converting other types to BF16.
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rd(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rn(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_ru(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rz(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rd(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rn(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_ru(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rz(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rd(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rn(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_ru(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rz(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rd(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rn(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_ru(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rz(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rd(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rn(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_ru(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rz(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rd(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rn(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_ru(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rz(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_double2bfloat16(double x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short_as_bfloat16(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort_as_bfloat16(unsigned short x);
}
#endif // __SYCL_DEVICE_ONLY__ && (defined(__SPIR__) || defined(__SPIRV__))

namespace sycl {

enum class rounding_mode { automatic = 0, rte = 1, rtz = 2, rtp = 3, rtn = 4 };

inline namespace _V1 {
#ifndef __SYCL_DEVICE_ONLY__
// TODO: Refactor includes so we can just "#include".
inline float ceil(float);
inline double ceil(double);
inline float floor(float);
inline double floor(double);
inline float rint(float);
inline double rint(double);
inline float trunc(float);
inline double trunc(double);
#endif
namespace detail {

template <typename FromT, typename ToT, sycl::rounding_mode RoundingMode,
          int VecSize, typename NativeFromT, typename NativeToT>
NativeToT convertImpl(NativeFromT);

template <typename T, typename R>
using is_sint_to_sint =
    std::bool_constant<is_sigeninteger_v<T> && is_sigeninteger_v<R>>;

template <typename T, typename R>
using is_uint_to_uint =
    std::bool_constant<is_sugeninteger_v<T> && is_sugeninteger_v<R>>;

template <typename T, typename R>
using is_sint_to_from_uint = std::bool_constant<
    (detail::is_sigeninteger_v<T> && detail::is_sugeninteger_v<R>) ||
    (detail::is_sugeninteger_v<T> && detail::is_sigeninteger_v<R>)>;

template <typename T, typename R>
using is_sint_to_float =
    std::bool_constant<std::is_integral_v<T> && !std::is_unsigned_v<T> &&
                       detail::is_floating_point<R>::value>;

template <typename T, typename R>
using is_uint_to_float =
    std::bool_constant<std::is_unsigned_v<T> &&
                       detail::is_floating_point<R>::value>;

template <typename T, typename R>
using is_int_to_float = std::bool_constant<std::is_integral_v<T> &&
                                           detail::is_floating_point<R>::value>;

template <typename T, typename R>
using is_float_to_uint =
    std::bool_constant<detail::is_floating_point<T>::value &&
                       std::is_unsigned_v<R>>;

template <typename T, typename R>
using is_float_to_sint =
    std::bool_constant<detail::is_floating_point<T>::value &&
                       std::is_integral_v<R> && !std::is_unsigned_v<R>>;

template <typename T, typename R>
using is_float_to_float =
    std::bool_constant<detail::is_floating_point<T>::value &&
                       detail::is_floating_point<R>::value>;

using bfloat16 = sycl::ext::oneapi::bfloat16;

#ifndef __SYCL_DEVICE_ONLY__
template <typename From, typename To, int VecSize,
          typename Enable = std::enable_if_t<VecSize == 1>>
To SConvert(From Value) {
  return static_cast<To>(Value);
}

template <typename From, typename To, int VecSize,
          typename Enable = std::enable_if_t<VecSize == 1>>
To UConvert(From Value) {
  return static_cast<To>(Value);
}

template <typename From, typename To, int VecSize,
          typename Enable = std::enable_if_t<VecSize == 1>>
To ConvertSToF(From Value) {
  return static_cast<To>(Value);
}

template <typename From, typename To, int VecSize,
          typename Enable = std::enable_if_t<VecSize == 1>>
To ConvertUToF(From Value) {
  return static_cast<To>(Value);
}

template <typename From, typename To, int VecSize,
          typename Enable = std::enable_if_t<VecSize == 1>,
          sycl::rounding_mode RM>
To FConvert(From Value) {
  return static_cast<To>(Value);
}

template <typename From, typename To, int VecSize,
          typename Enable = std::enable_if_t<VecSize == 1>,
          sycl::rounding_mode roundingMode>
To ConvertFToS(From Value) {
  switch (roundingMode) {
    // Round to nearest even is default rounding mode for floating-point types
  case rounding_mode::automatic:
    // Round to nearest even.
  case rounding_mode::rte: {
    int OldRoundingDirection = std::fegetround();
    int Err = std::fesetround(FE_TONEAREST);
    if (Err)
      throw sycl::exception(make_error_code(errc::runtime),
                            "Unable to set rounding mode to FE_TONEAREST");
    To Result = sycl::rint(Value);
    Err = std::fesetround(OldRoundingDirection);
    if (Err)
      throw sycl::exception(make_error_code(errc::runtime),
                            "Unable to restore rounding mode.");
    return Result;
  }
    // Round toward zero.
  case rounding_mode::rtz:
    return sycl::trunc(Value);
    // Round toward positive infinity.
  case rounding_mode::rtp:
    return sycl::ceil(Value);
    // Round toward negative infinity.
  case rounding_mode::rtn:
    return sycl::floor(Value);
  };
  assert(false && "Unsupported rounding mode!");
  return static_cast<To>(Value);
}

template <typename From, typename To, int VecSize,
          typename Enable = std::enable_if_t<VecSize == 1>,
          sycl::rounding_mode roundingMode>
To ConvertFToU(From Value) {
  return ConvertFToS<From, To, VecSize, Enable, roundingMode>(Value);
}

template <typename NativeToT, sycl::rounding_mode RoundingMode>
inline NativeToT ConvertFromBF16Scalar(bfloat16 val) {
  // On host, NativeBF16T is bfloat16. Convert BF16 to float losslessly.
  float fval = static_cast<float>(val);

  if constexpr (std::is_same_v<NativeToT, float>)
    return fval;
  else
    // Convert float to the desired type.
    return convertImpl<float, NativeToT, RoundingMode, 1, float, NativeToT>(
        fval);
}

template <typename NativeFromT, sycl::rounding_mode RoundingMode>
bfloat16 ConvertToBF16Scalar(NativeFromT val) {

  constexpr int rm = static_cast<int>(RoundingMode);
  return sycl::ext::oneapi::detail::ConvertToBfloat16::
      getBfloat16WithRoundingMode<NativeFromT, rm>(val);
}

#else
// Bunch of helpers to "specialize" each template for its own destination type
// and vector size.

// Added for unification, to be able to have single enable_if-like trait for all
// cases regardless of whether rounding mode is actually applicable or not.
template <rounding_mode Mode> using AnyRM = std::bool_constant<true>;

template <rounding_mode Mode>
using RteOrAutomatic = std::bool_constant<Mode == rounding_mode::automatic ||
                                          Mode == rounding_mode::rte>;

template <rounding_mode Mode>
using Rtz = std::bool_constant<Mode == rounding_mode::rtz>;

template <rounding_mode Mode>
using Rtp = std::bool_constant<Mode == rounding_mode::rtp>;

template <rounding_mode Mode>
using Rtn = std::bool_constant<Mode == rounding_mode::rtn>;

template <int VecSize> using IsScalar = std::bool_constant<VecSize == 1>;

template <int ExpectedVecSize, int ActualVecSize>
using IsVectorOf = std::bool_constant<ActualVecSize == ExpectedVecSize>;

// This is a key condition for "specializations" below: it helps restrict each
// "specialization" to a (mostly) single type with one exception for
// signed char -> char case.
template <typename ExpectedType, typename ActualType>
using IsExpectedIntType =
    std::bool_constant<std::is_same_v<ExpectedType, ActualType> ||
                       (std::is_same_v<ExpectedType, sycl::opencl::cl_char> &&
                        std::is_same_v<ActualType, signed char>)>;

// Helpers which are used for conversions to an integer type
template <typename ExpectedType, typename ActualType, int VecSize,
          typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition,
          sycl::rounding_mode RoundingMode>
struct enable_if_to_int_scalar
    : std::enable_if<IsExpectedIntType<ExpectedType, ActualType>::value &&
                         IsScalar<VecSize>::value &&
                         RoundingModeCondition<RoundingMode>::value,
                     ReturnType> {};

template <typename ExpectedType, typename ActualType, int VecSize,
          typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition = AnyRM,
          sycl::rounding_mode RoundingMode = sycl::rounding_mode::automatic>
using enable_if_to_int_scalar_t =
    typename enable_if_to_int_scalar<ExpectedType, ActualType, VecSize,
                                     ReturnType, RoundingModeCondition,
                                     RoundingMode>::type;

template <typename ExpectedType, typename ActualType, int ExpectedVecSize,
          int ActualVecSize, typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition,
          sycl::rounding_mode RoundingMode>
struct enable_if_to_int_vector
    : std::enable_if<IsExpectedIntType<ExpectedType, ActualType>::value &&
                         IsVectorOf<ExpectedVecSize, ActualVecSize>::value &&
                         RoundingModeCondition<RoundingMode>::value,
                     ReturnType> {};

template <typename ExpectedType, typename ActualType, int ExpectedVecSize,
          int ActualVecSize, typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition = AnyRM,
          sycl::rounding_mode RoundingMode = sycl::rounding_mode::automatic>
using enable_if_to_int_vector_t =
    typename enable_if_to_int_vector<ExpectedType, ActualType, ExpectedVecSize,
                                     ActualVecSize, ReturnType,
                                     RoundingModeCondition, RoundingMode>::type;

// signed to signed, unsigned to unsigned conversions
#define __SYCL_SCALAR_INT_INT_CONVERT(Op, DestType)                            \
  template <typename From, typename To, int VecSize, typename Enable>          \
  enable_if_to_int_scalar_t<sycl::opencl::cl_##DestType, Enable, VecSize, To>  \
      Op##Convert(From value) {                                                \
    return __spirv_##Op##Convert_R##DestType(value);                           \
  }

#define __SYCL_VECTOR_INT_INT_CONVERT(Op, N, DestType)                         \
  template <typename From, typename To, int VecSize, typename Enable>          \
  enable_if_to_int_vector_t<sycl::opencl::cl_##DestType, Enable, N, VecSize,   \
                            To>                                                \
      Op##Convert(From value) {                                                \
    return __spirv_##Op##Convert_R##DestType##N(value);                        \
  }

#define __SYCL_INT_INT_CONVERT(Op, DestType)                                   \
  __SYCL_SCALAR_INT_INT_CONVERT(Op, DestType)                                  \
  __SYCL_VECTOR_INT_INT_CONVERT(Op, 2, DestType)                               \
  __SYCL_VECTOR_INT_INT_CONVERT(Op, 3, DestType)                               \
  __SYCL_VECTOR_INT_INT_CONVERT(Op, 4, DestType)                               \
  __SYCL_VECTOR_INT_INT_CONVERT(Op, 8, DestType)                               \
  __SYCL_VECTOR_INT_INT_CONVERT(Op, 16, DestType)

__SYCL_INT_INT_CONVERT(S, char)
__SYCL_INT_INT_CONVERT(S, short)
__SYCL_INT_INT_CONVERT(S, int)
__SYCL_INT_INT_CONVERT(S, long)

__SYCL_INT_INT_CONVERT(U, uchar)
__SYCL_INT_INT_CONVERT(U, ushort)
__SYCL_INT_INT_CONVERT(U, uint)
__SYCL_INT_INT_CONVERT(U, ulong)

#undef __SYCL_SCALAR_INT_INT_CONVERT
#undef __SYCL_VECTOR_INT_INT_CONVERT
#undef __SYCL_INT_INT_CONVERT

// float to signed, float to unsigned conversion
#define __SYCL_SCALAR_FLOAT_INT_CONVERT(Op, DestType, RoundingMode,            \
                                        RoundingModeCondition)                 \
  template <typename From, typename To, int VecSize, typename Enable,          \
            sycl::rounding_mode RM>                                            \
  enable_if_to_int_scalar_t<sycl::opencl::cl_##DestType, Enable, VecSize, To,  \
                            RoundingModeCondition, RM>                         \
      Convert##Op(From Value) {                                                \
    return __spirv_Convert##Op##_R##DestType##_##RoundingMode(Value);          \
  }

#define __SYCL_VECTOR_FLOAT_INT_CONVERT(Op, N, DestType, RoundingMode,         \
                                        RoundingModeCondition)                 \
  template <typename From, typename To, int VecSize, typename Enable,          \
            sycl::rounding_mode RM>                                            \
  enable_if_to_int_vector_t<sycl::opencl::cl_##DestType, Enable, N, VecSize,   \
                            To, RoundingModeCondition, RM>                     \
      Convert##Op(From Value) {                                                \
    return __spirv_Convert##Op##_R##DestType##N##_##RoundingMode(Value);       \
  }

#define __SYCL_FLOAT_INT_CONVERT(Op, DestType, RoundingMode,                   \
                                 RoundingModeCondition)                        \
  __SYCL_SCALAR_FLOAT_INT_CONVERT(Op, DestType, RoundingMode,                  \
                                  RoundingModeCondition)                       \
  __SYCL_VECTOR_FLOAT_INT_CONVERT(Op, 2, DestType, RoundingMode,               \
                                  RoundingModeCondition)                       \
  __SYCL_VECTOR_FLOAT_INT_CONVERT(Op, 3, DestType, RoundingMode,               \
                                  RoundingModeCondition)                       \
  __SYCL_VECTOR_FLOAT_INT_CONVERT(Op, 4, DestType, RoundingMode,               \
                                  RoundingModeCondition)                       \
  __SYCL_VECTOR_FLOAT_INT_CONVERT(Op, 8, DestType, RoundingMode,               \
                                  RoundingModeCondition)                       \
  __SYCL_VECTOR_FLOAT_INT_CONVERT(Op, 16, DestType, RoundingMode,              \
                                  RoundingModeCondition)

#define __SYCL_FLOAT_INT_CONVERT_FOR_TYPE(Op, DestType)                        \
  __SYCL_FLOAT_INT_CONVERT(Op, DestType, rte, RteOrAutomatic)                  \
  __SYCL_FLOAT_INT_CONVERT(Op, DestType, rtz, Rtz)                             \
  __SYCL_FLOAT_INT_CONVERT(Op, DestType, rtp, Rtp)                             \
  __SYCL_FLOAT_INT_CONVERT(Op, DestType, rtn, Rtn)

__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToS, char)
__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToS, short)
__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToS, int)
__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToS, long)

__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToU, uchar)
__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToU, ushort)
__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToU, uint)
__SYCL_FLOAT_INT_CONVERT_FOR_TYPE(FToU, ulong)

#undef __SYCL_SCALAR_FLOAT_INT_CONVERT
#undef __SYCL_VECTOR_FLOAT_INT_CONVERT
#undef __SYCL_FLOAT_INT_CONVERT
#undef __SYCL_FLOAT_INT_CONVERT_FOR_TYPE

// Helpers which are used for conversions to a floating-point type
template <typename ExpectedType, typename ActualType>
using IsExpectedFloatType =
    std::bool_constant<std::is_same_v<ExpectedType, ActualType> ||
                       (std::is_same_v<ExpectedType, sycl::opencl::cl_half> &&
                        std::is_same_v<ActualType, _Float16>)>;

template <typename ExpectedType, typename ActualType, int VecSize,
          typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition,
          sycl::rounding_mode RoundingMode>
struct enable_if_to_float_scalar
    : std::enable_if<IsExpectedFloatType<ExpectedType, ActualType>::value &&
                         IsScalar<VecSize>::value &&
                         RoundingModeCondition<RoundingMode>::value,
                     ReturnType> {};

template <typename ExpectedType, typename ActualType, int VecSize,
          typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition = AnyRM,
          sycl::rounding_mode RoundingMode = sycl::rounding_mode::automatic>
using enable_if_to_float_scalar_t =
    typename enable_if_to_float_scalar<ExpectedType, ActualType, VecSize,
                                       ReturnType, RoundingModeCondition,
                                       RoundingMode>::type;

template <typename ExpectedType, typename ActualType, int ExpectedVecSize,
          int ActualVecSize, typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition,
          sycl::rounding_mode RoundingMode>
struct enable_if_to_float_vector
    : std::enable_if<IsExpectedFloatType<ExpectedType, ActualType>::value &&
                         IsVectorOf<ExpectedVecSize, ActualVecSize>::value &&
                         RoundingModeCondition<RoundingMode>::value,
                     ReturnType> {};

template <typename ExpectedType, typename ActualType, int ExpectedVecSize,
          int ActualVecSize, typename ReturnType,
          template <sycl::rounding_mode> typename RoundingModeCondition = AnyRM,
          sycl::rounding_mode RoundingMode = sycl::rounding_mode::automatic>
using enable_if_to_float_vector_t = typename enable_if_to_float_vector<
    ExpectedType, ActualType, ExpectedVecSize, ActualVecSize, ReturnType,
    RoundingModeCondition, RoundingMode>::type;

// signed to float, unsigned to float conversions
#define __SYCL_SCALAR_INT_FLOAT_CONVERT(Op, DestType)                          \
  template <typename From, typename To, int VecSize, typename Enable>          \
  enable_if_to_float_scalar_t<sycl::opencl::cl_##DestType, Enable, VecSize,    \
                              To>                                              \
      Convert##Op(From value) {                                                \
    return __spirv_Convert##Op##_R##DestType(value);                           \
  }

#define __SYCL_VECTOR_INT_FLOAT_CONVERT(Op, N, DestType)                       \
  template <typename From, typename To, int VecSize, typename Enable>          \
  enable_if_to_float_vector_t<sycl::opencl::cl_##DestType, Enable, N, VecSize, \
                              To>                                              \
      Convert##Op(From value) {                                                \
    return __spirv_Convert##Op##_R##DestType##N(value);                        \
  }

#define __SYCL_INT_FLOAT_CONVERT(Op, DestType)                                 \
  __SYCL_SCALAR_INT_FLOAT_CONVERT(Op, DestType)                                \
  __SYCL_VECTOR_INT_FLOAT_CONVERT(Op, 2, DestType)                             \
  __SYCL_VECTOR_INT_FLOAT_CONVERT(Op, 3, DestType)                             \
  __SYCL_VECTOR_INT_FLOAT_CONVERT(Op, 4, DestType)                             \
  __SYCL_VECTOR_INT_FLOAT_CONVERT(Op, 8, DestType)                             \
  __SYCL_VECTOR_INT_FLOAT_CONVERT(Op, 16, DestType)

__SYCL_INT_FLOAT_CONVERT(SToF, half)
__SYCL_INT_FLOAT_CONVERT(SToF, float)
__SYCL_INT_FLOAT_CONVERT(SToF, double)

__SYCL_INT_FLOAT_CONVERT(UToF, half)
__SYCL_INT_FLOAT_CONVERT(UToF, float)
__SYCL_INT_FLOAT_CONVERT(UToF, double)

#undef __SYCL_SCALAR_INT_FLOAT_CONVERT
#undef __SYCL_VECTOR_INT_FLOAT_CONVERT
#undef __SYCL_INT_FLOAT_CONVERT

// float to float conversions
#define __SYCL_SCALAR_FLOAT_FLOAT_CONVERT(DestType, RoundingMode,              \
                                          RoundingModeCondition)               \
  template <typename From, typename To, int VecSize, typename Enable,          \
            sycl::rounding_mode RM>                                            \
  enable_if_to_float_scalar_t<sycl::opencl::cl_##DestType, Enable, VecSize,    \
                              To, RoundingModeCondition, RM>                   \
  FConvert(From Value) {                                                       \
    return __spirv_FConvert_R##DestType##_##RoundingMode(Value);               \
  }

#define __SYCL_VECTOR_FLOAT_FLOAT_CONVERT(N, DestType, RoundingMode,           \
                                          RoundingModeCondition)               \
  template <typename From, typename To, int VecSize, typename Enable,          \
            sycl::rounding_mode RM>                                            \
  enable_if_to_float_vector_t<sycl::opencl::cl_##DestType, Enable, N, VecSize, \
                              To, RoundingModeCondition, RM>                   \
  FConvert(From Value) {                                                       \
    return __spirv_FConvert_R##DestType##N##_##RoundingMode(Value);            \
  }

#define __SYCL_FLOAT_FLOAT_CONVERT(DestType, RoundingMode,                     \
                                   RoundingModeCondition)                      \
  __SYCL_SCALAR_FLOAT_FLOAT_CONVERT(DestType, RoundingMode,                    \
                                    RoundingModeCondition)                     \
  __SYCL_VECTOR_FLOAT_FLOAT_CONVERT(2, DestType, RoundingMode,                 \
                                    RoundingModeCondition)                     \
  __SYCL_VECTOR_FLOAT_FLOAT_CONVERT(3, DestType, RoundingMode,                 \
                                    RoundingModeCondition)                     \
  __SYCL_VECTOR_FLOAT_FLOAT_CONVERT(4, DestType, RoundingMode,                 \
                                    RoundingModeCondition)                     \
  __SYCL_VECTOR_FLOAT_FLOAT_CONVERT(8, DestType, RoundingMode,                 \
                                    RoundingModeCondition)                     \
  __SYCL_VECTOR_FLOAT_FLOAT_CONVERT(16, DestType, RoundingMode,                \
                                    RoundingModeCondition)

#define __SYCL_FLOAT_FLOAT_CONVERT_FOR_TYPE(DestType)                          \
  __SYCL_FLOAT_FLOAT_CONVERT(DestType, rte, RteOrAutomatic)                    \
  __SYCL_FLOAT_FLOAT_CONVERT(DestType, rtz, Rtz)                               \
  __SYCL_FLOAT_FLOAT_CONVERT(DestType, rtp, Rtp)                               \
  __SYCL_FLOAT_FLOAT_CONVERT(DestType, rtn, Rtn)

__SYCL_FLOAT_FLOAT_CONVERT_FOR_TYPE(half)
__SYCL_FLOAT_FLOAT_CONVERT_FOR_TYPE(float)
__SYCL_FLOAT_FLOAT_CONVERT_FOR_TYPE(double)

#undef __SYCL_SCALAR_FLOAT_FLOAT_CONVERT
#undef __SYCL_VECTOR_FLOAT_FLOAT_CONVERT
#undef __SYCL_FLOAT_FLOAT_CONVERT
#undef __SYCL_FLOAT_FLOAT_CONVERT_FOR_TYPE

template <typename NativeBFT, typename NativeFloatT, int VecSize>
inline NativeFloatT ConvertBF16ToFVec(NativeBFT vec) {
  bfloat16 *src = sycl::bit_cast<bfloat16 *>(&vec);

  // OpenCL vector of 3 elements is aligned to 4 multiplied by
  // the size of data type.
  constexpr int AdjustedSize = (VecSize == 3) ? 4 : VecSize;
  float dst[AdjustedSize];
  sycl::ext::oneapi::detail::BF16VecToFloatVec<VecSize>(src, dst);

  return sycl::bit_cast<NativeFloatT>(dst);
}

template <typename NativeFloatT, typename NativeBFT, int VecSize>
inline NativeBFT ConvertFToBF16Vec(NativeFloatT vec) {
  float *src = sycl::bit_cast<float *>(&vec);

  // OpenCL vector of 3 elements is aligned to 4 multiplied by
  // the size of data type.
  constexpr int AdjustedSize = (VecSize == 3) ? 4 : VecSize;
  bfloat16 dst[AdjustedSize];

  sycl::ext::oneapi::detail::FloatVecToBF16Vec<VecSize>(src, dst);
  return sycl::bit_cast<NativeBFT>(dst);
}

/* Emit _imf_* funcs only on Intel hardware.  */
#if defined(__SPIR__) || defined(__SPIRV__)
#define EXPAND_BF16_ROUNDING_MODE(type, type_str, rmode, rmode_str)            \
  template <typename NativeToT, sycl::rounding_mode RoundingMode>              \
  std::enable_if_t<(std::is_same_v<NativeToT, type> && RoundingMode == rmode), \
                   NativeToT>                                                  \
  ConvertFromBF16Scalar(uint16_t val) {                                        \
    return __imf_bfloat162##type_str##_##rmode_str(val);                       \
  }                                                                            \
  template <typename NativeFromT, sycl::rounding_mode RoundingMode>            \
  std::enable_if_t<                                                            \
      (std::is_same_v<NativeFromT, type> && RoundingMode == rmode), uint16_t>  \
  ConvertToBF16Scalar(NativeFromT val) {                                       \
    return __imf_##type_str##2bfloat16_##rmode_str(val);                       \
  }

#else // __SYCL_DEVICE_ONLY__ && (defined(__SPIR__) || defined(__SPIRV__))
// On non-Intel HWs, convert BF16 to float (losslessly) and convert float
// to the desired type.
#define EXPAND_BF16_ROUNDING_MODE(type, type_str, rmode, rmode_str)            \
  template <typename NativeToT, sycl::rounding_mode RoundingMode>              \
  std::enable_if_t<(std::is_same_v<NativeToT, type> && RoundingMode == rmode), \
                   NativeToT>                                                  \
  ConvertFromBF16Scalar(uint16_t val) {                                        \
    bfloat16 bfval = sycl::bit_cast<bfloat16>(val);                            \
    float fval = static_cast<float>(bfval);                                    \
    return convertImpl<fval, NativeToT, RoundingMode, 1, float, NativeToT>(    \
        fval);                                                                 \
  }                                                                            \
  template <typename NativeFromT, sycl::rounding_mode RoundingMode>            \
  std::enable_if_t<                                                            \
      (std::is_same_v<NativeFromT, type> && RoundingMode == rmode), uint16_t>  \
  ConvertToBF16Scalar(NativeFromT val) {                                       \
    constexpr int rm = static_cast<int>(RoundingMode);                         \
    bfloat16 bfval = sycl::ext::oneapi::detail::ConvertToBfloat16::            \
        getBfloat16WithRoundingMode<NativeFromT, rm>(val);                     \
    return sycl::bit_cast<uint16_t>(bfval);                                    \
  }
#endif // __SYCL_DEVICE_ONLY__ && (defined(__SPIR__) || defined(__SPIRV__))

#define EXPAND_BF16_TYPE(type, type_str)                                       \
  EXPAND_BF16_ROUNDING_MODE(type, type_str, sycl::rounding_mode::automatic,    \
                            rn)                                                \
  EXPAND_BF16_ROUNDING_MODE(type, type_str, sycl::rounding_mode::rte, rn)      \
  EXPAND_BF16_ROUNDING_MODE(type, type_str, sycl::rounding_mode::rtp, ru)      \
  EXPAND_BF16_ROUNDING_MODE(type, type_str, sycl::rounding_mode::rtn, rd)      \
  EXPAND_BF16_ROUNDING_MODE(type, type_str, sycl::rounding_mode::rtz, rz)

EXPAND_BF16_TYPE(uint, uint)
EXPAND_BF16_TYPE(int, int)
EXPAND_BF16_TYPE(ushort, ushort)
EXPAND_BF16_TYPE(short, short)
EXPAND_BF16_TYPE(long, ll)
EXPAND_BF16_TYPE(unsigned long long, ull)

#undef EXPAND_BF16_TYPE
#undef EXPAND_BF16_ROUNDING_MODE

// Mapping from BF16 to float is 1:1, lossless, so we accept all
// rounding modes.
template <typename NativeToT, sycl::rounding_mode RoundingMode>
std::enable_if_t<std::is_same_v<NativeToT, float>, NativeToT>
ConvertFromBF16Scalar(uint16_t val) {
  bfloat16 bfval = sycl::bit_cast<bfloat16>(val);
  return static_cast<float>(bfval);
}

template <typename NativeFromT, sycl::rounding_mode RoundingMode>
std::enable_if_t<std::is_same_v<NativeFromT, double>, uint16_t>
ConvertToBF16Scalar(NativeFromT val) {
#if defined(__SPIR__) || defined(__SPIRV__)
  return __imf_double2bfloat16(val);
#else
  constexpr int rm = static_cast<int>(RoundingMode);
  bfloat16 bfval =
      sycl::ext::oneapi::detail::ConvertToBfloat16::getBfloat16WithRoundingMode<
          NativeFromT, rm>(val);
  return sycl::bit_cast<uint16_t>(bfval);
#endif
}

template <typename NativeFromT, sycl::rounding_mode RoundingMode>
std::enable_if_t<std::is_same_v<NativeFromT, float>, uint16_t>
ConvertToBF16Scalar(NativeFromT val) {

#if defined(__SPIR__) || defined(__SPIRV__)
  if constexpr (RoundingMode == sycl::rounding_mode::automatic ||
                RoundingMode == sycl::rounding_mode::rte)
    return __imf_float2bfloat16_rn(val);
  else if constexpr (RoundingMode == sycl::rounding_mode::rtp)
    return __imf_float2bfloat16_ru(val);
  else if constexpr (RoundingMode == sycl::rounding_mode::rtn)
    return __imf_float2bfloat16_rd(val);
  else if constexpr (RoundingMode == sycl::rounding_mode::rtz)
    return __imf_float2bfloat16_rz(val);
  else
    static_assert(false, "Invalid rounding mode.");
#else
  constexpr int rm = static_cast<int>(RoundingMode);
  bfloat16 bfval =
      sycl::ext::oneapi::detail::ConvertToBfloat16::getBfloat16WithRoundingMode<
          float, rm>(val);
  return sycl::bit_cast<uint16_t>(bfval);
#endif
}

#endif // __SYCL_DEVICE_ONLY__

// Wrapper function for scalar and vector conversions from BF16 type.
template <typename ToT, typename NativeFromT, typename NativeToT,
          sycl::rounding_mode RoundingMode, int VecSize>
NativeToT ConvertFromBF16(NativeFromT val) {
#ifdef __SYCL_DEVICE_ONLY__
  //  Use vector conversion from BF16 to float for all rounding modes.
  if constexpr (std::is_same_v<ToT, float> && VecSize > 1)
    return ConvertBF16ToFVec<NativeFromT, NativeToT, VecSize>(val);
  else
#endif
    // For VecSize > 1. Only for device.
    if constexpr (VecSize > 1) {
      NativeToT retval;
      for (int i = 0; i < VecSize; i++) {
        retval[i] = ConvertFromBF16Scalar<ToT, RoundingMode>(val[i]);
      }
      return retval;
    }
    // For VecSize == 1.
    else
      return ConvertFromBF16Scalar<NativeToT, RoundingMode>(val);
}

// Wrapper function for scalar and vector conversions to BF16 type.
template <typename FromT, typename NativeFromT, typename NativeToT,
          sycl::rounding_mode RoundingMode, int VecSize>
NativeToT ConvertToBF16(NativeFromT val) {
#ifdef __SYCL_DEVICE_ONLY__
  //  Use vector conversion to BF16 from float for RNE rounding mode.
  if constexpr (std::is_same_v<FromT, float> && VecSize > 1 &&
                (RoundingMode == sycl::rounding_mode::automatic ||
                 RoundingMode == sycl::rounding_mode::rte))
    return ConvertFToBF16Vec<NativeFromT, NativeToT, VecSize>(val);
  else
#endif
    // For VecSize > 1. Only for device.
    if constexpr (VecSize > 1) {
      NativeToT retval;
      for (int i = 0; i < VecSize; i++) {
        retval[i] = ConvertToBF16Scalar<FromT, RoundingMode>(val[i]);
      }
      return retval;
    }
    // For VecSize == 1.
    else
      return ConvertToBF16Scalar<NativeFromT, RoundingMode>(val);
}

/// Entry point helper for all kinds of converts between scalars and vectors, it
/// dispatches to a right function depending on source and destination types.
///
/// \tparam FromT \b scalar user-visible type to convert \a from, used to detect
/// conversion kind. It is expected to be \c DataT template argument of a vector
/// we are trying to convert \a from
/// \tparam ToT \b scalar user-visible type to convert \a to, used to detect
/// conversion kind. It is expected to be \c DataT template argument of a vector
/// we are trying to convert \a to
/// \tparam NativeFromT \b scalar or \b vector internal type corresponding to
/// \c FromT, which is used to hold vector data. It is expected to be
/// vec<FromT, VecSize>::vector_t of a vector we are trying to convert \a from
/// if VecSize > 1, or result of detail::ConvertToOpenCLType_t<FromT>
/// \tparam NativeToT \b scalar or \b vector internal type corresponding to
/// \c ToT, which is used to hold vector data. It is expected to be
/// vec<ToT, VecSize>::vector_t of a vector we are trying to convert \a from
/// if VecSize > 1, or result of detail::ConvertToOpenCLType_t<ToT>
///
/// \note Each pair of types FromT, ToT and NativeFromT, NativeToT can't contain
/// the same type, because there are no no-op convert instructions in SPIR-V.
template <typename FromT, typename ToT, sycl::rounding_mode RoundingMode,
          int VecSize, typename NativeFromT, typename NativeToT>
NativeToT convertImpl(NativeFromT Value) {
  static_assert(!std::is_same_v<FromT, ToT>);
  static_assert(!std::is_same_v<NativeFromT, NativeToT>);
  using ElemTy = typename detail::ConvertToOpenCLType_t<ToT>;
  if constexpr (is_sint_to_sint<FromT, ToT>::value)
    return SConvert<NativeFromT, NativeToT, VecSize, ElemTy>(Value);
  else if constexpr (is_uint_to_uint<FromT, ToT>::value)
    return UConvert<NativeFromT, NativeToT, VecSize, ElemTy>(Value);
  else if constexpr (is_sint_to_float<FromT, ToT>::value)
    return ConvertSToF<NativeFromT, NativeToT, VecSize, ElemTy>(Value);
  else if constexpr (is_uint_to_float<FromT, ToT>::value)
    return ConvertUToF<NativeFromT, NativeToT, VecSize, ElemTy>(Value);
  else if constexpr (is_float_to_float<FromT, ToT>::value)
    return FConvert<NativeFromT, NativeToT, VecSize, ElemTy, RoundingMode>(
        Value);
  // BF16 conversion to other types.
  else if constexpr (std::is_same_v<FromT, bfloat16>)
    return ConvertFromBF16<ToT, NativeFromT, NativeToT, RoundingMode, VecSize>(
        Value);
  // conversion from other types to BF16.
  else if constexpr (std::is_same_v<ToT, bfloat16>)
    return ConvertToBF16<FromT, NativeFromT, NativeToT, RoundingMode, VecSize>(
        Value);
  else if constexpr (is_float_to_sint<FromT, ToT>::value)
    return ConvertFToS<NativeFromT, NativeToT, VecSize, ElemTy, RoundingMode>(
        Value);
  else if constexpr (is_float_to_uint<FromT, ToT>::value)
    return ConvertFToU<NativeFromT, NativeToT, VecSize, ElemTy, RoundingMode>(
        Value);
  else {
    static_assert(is_sint_to_from_uint<FromT, ToT>::value,
                  "Unexpected conversion type");
    static_assert(VecSize == 1, "Conversion between signed and unsigned data "
                                "types is only available for scalars");
    // vec::convert is underspecified and therefore it is not entirely clear
    // what to do here. 'static_cast' implementation matches SYCL CTS and it
    // matches our old implementation. Unfortunately, OpSetConvertUToS and
    // OpSatConvertSToU behave differently and we can't use them here until the
    // behavior of conversions is well-defined by the SYCL 2020 specificiation.
    // See https://github.com/KhronosGroup/SYCL-Docs/issues/492
    return static_cast<NativeToT>(Value);
  }
}

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename FromT, typename ToT, sycl::rounding_mode RoundingMode,
          int VecSize, typename NativeFromT, typename NativeToT>
auto ConvertImpl(std::byte val) {
  return convertImpl<FromT, ToT, RoundingMode, VecSize, NativeFromT, NativeToT>(
      (std::int8_t)val);
}
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl
