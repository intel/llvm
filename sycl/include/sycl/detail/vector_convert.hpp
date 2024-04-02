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

#ifndef __SYCL_DEVICE_ONLY__
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/builtins_legacy_scalar.hpp> // for ceil, floor, rint, trunc
#endif
#include <cfenv> // for fesetround, fegetround
#endif

#include <type_traits>

namespace sycl {

enum class rounding_mode { automatic = 0, rte = 1, rtz = 2, rtp = 3, rtn = 4 };

inline namespace _V1 {
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
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
#endif
namespace detail {

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

#endif // __SYCL_DEVICE_ONLY__

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

} // namespace detail
} // namespace _V1
} // namespace sycl
