//==-- vector_convert.hpp --- vec::convert implementation ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...

#ifndef __SYCL_DEVICE_ONLY__

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/builtins_scalar_gen.hpp> // for ceil, floor, rint, trunc
#else                                   // __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/builtins_legacy_scalar.hpp> // for ceil, floor, rint, trunc
#endif                                     // __INTEL_PREVIEW_BREAKING_CHANGES

#include <cfenv> // for fesetround, fegetround
#endif

#include <type_traits>

namespace sycl {

enum class rounding_mode { automatic = 0, rte = 1, rtz = 2, rtp = 3, rtn = 4 };

inline namespace _V1 {
namespace detail {

template <typename T, typename R>
using is_int_to_int = std::integral_constant<bool, std::is_integral_v<T> &&
                                                       std::is_integral_v<R>>;

template <typename T, typename R>
using is_sint_to_sint =
    std::integral_constant<bool, is_sigeninteger_v<T> && is_sigeninteger_v<R>>;

template <typename T, typename R>
using is_uint_to_uint =
    std::integral_constant<bool, is_sugeninteger_v<T> && is_sugeninteger_v<R>>;

template <typename T, typename R>
using is_sint_to_from_uint =
    std::integral_constant<bool,
                           (is_sugeninteger_v<T> && is_sigeninteger_v<R>) ||
                               (is_sigeninteger_v<T> && is_sugeninteger_v<R>)>;

template <typename T, typename R>
using is_sint_to_float = std::integral_constant<
    bool, std::is_integral_v<T> &&
              !(std::is_unsigned_v<T>)&&detail::is_floating_point<R>::value>;

template <typename T, typename R>
using is_uint_to_float =
    std::integral_constant<bool, std::is_unsigned_v<T> &&
                                     detail::is_floating_point<R>::value>;

template <typename T, typename R>
using is_int_to_float =
    std::integral_constant<bool, std::is_integral_v<T> &&
                                     detail::is_floating_point<R>::value>;

template <typename T, typename R>
using is_float_to_int =
    std::integral_constant<bool, detail::is_floating_point<T>::value &&
                                     std::is_integral_v<R>>;

template <typename T, typename R>
using is_float_to_float =
    std::integral_constant<bool, detail::is_floating_point<T>::value &&
                                     detail::is_floating_point<R>::value>;
template <typename T>
using is_standard_type = std::integral_constant<bool, detail::is_sgentype_v<T>>;

template <typename T, typename R, rounding_mode roundingMode, typename OpenCLT,
          typename OpenCLR>
std::enable_if_t<std::is_same_v<T, R>, R> convertImpl(T Value) {
  return Value;
}

#ifndef __SYCL_DEVICE_ONLY__

// Note for float to half conversions, static_cast calls the conversion operator
// implemented for host that takes care of the precision requirements.
template <typename T, typename R, rounding_mode roundingMode, typename OpenCLT,
          typename OpenCLR>
std::enable_if_t<!std::is_same_v<T, R> && (is_int_to_int<T, R>::value ||
                                           is_int_to_float<T, R>::value ||
                                           is_float_to_float<T, R>::value),
                 R>
convertImpl(T Value) {
  return static_cast<R>(Value);
}

// float to int
template <typename T, typename R, rounding_mode roundingMode, typename OpenCLT,
          typename OpenCLR>
std::enable_if_t<is_float_to_int<T, R>::value, R> convertImpl(T Value) {
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
    R Result = sycl::rint(Value);
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
  return static_cast<R>(Value);
}
#else

template <rounding_mode Mode>
using RteOrAutomatic = std::bool_constant<Mode == rounding_mode::automatic ||
                                          Mode == rounding_mode::rte>;

template <rounding_mode Mode>
using Rtz = std::bool_constant<Mode == rounding_mode::rtz>;

template <rounding_mode Mode>
using Rtp = std::bool_constant<Mode == rounding_mode::rtp>;

template <rounding_mode Mode>
using Rtn = std::bool_constant<Mode == rounding_mode::rtn>;

// convert types with an equal size and diff names
template <typename T, typename R, rounding_mode roundingMode, typename OpenCLT,
          typename OpenCLR>
std::enable_if_t<!std::is_same_v<T, R> && std::is_same_v<OpenCLT, OpenCLR>, R>
convertImpl(T Value) {
  return static_cast<R>(Value);
}

// signed to signed
#define __SYCL_GENERATE_CONVERT_IMPL(DestType)                                 \
  template <typename T, typename R, rounding_mode roundingMode,                \
            typename OpenCLT, typename OpenCLR>                                \
  std::enable_if_t<is_sint_to_sint<T, R>::value &&                             \
                       !std::is_same_v<OpenCLT, OpenCLR> &&                    \
                       (std::is_same_v<OpenCLR, opencl::cl_##DestType> ||      \
                        (std::is_same_v<OpenCLR, signed char> &&               \
                         std::is_same_v<DestType, char>)),                     \
                   R>                                                          \
  convertImpl(T Value) {                                                       \
    OpenCLT OpValue = sycl::detail::convertDataToType<T, OpenCLT>(Value);      \
    return __spirv_SConvert##_R##DestType(OpValue);                            \
  }

__SYCL_GENERATE_CONVERT_IMPL(char)
__SYCL_GENERATE_CONVERT_IMPL(short)
__SYCL_GENERATE_CONVERT_IMPL(int)
__SYCL_GENERATE_CONVERT_IMPL(long)

#undef __SYCL_GENERATE_CONVERT_IMPL

// unsigned to unsigned
#define __SYCL_GENERATE_CONVERT_IMPL(DestType)                                 \
  template <typename T, typename R, rounding_mode roundingMode,                \
            typename OpenCLT, typename OpenCLR>                                \
  std::enable_if_t<is_uint_to_uint<T, R>::value &&                             \
                       !std::is_same_v<OpenCLT, OpenCLR> &&                    \
                       std::is_same_v<OpenCLR, opencl::cl_##DestType>,         \
                   R>                                                          \
  convertImpl(T Value) {                                                       \
    OpenCLT OpValue = sycl::detail::convertDataToType<T, OpenCLT>(Value);      \
    return __spirv_UConvert##_R##DestType(OpValue);                            \
  }

__SYCL_GENERATE_CONVERT_IMPL(uchar)
__SYCL_GENERATE_CONVERT_IMPL(ushort)
__SYCL_GENERATE_CONVERT_IMPL(uint)
__SYCL_GENERATE_CONVERT_IMPL(ulong)

#undef __SYCL_GENERATE_CONVERT_IMPL

// unsigned to (from) signed
template <typename T, typename R, rounding_mode roundingMode, typename OpenCLT,
          typename OpenCLR>
std::enable_if_t<is_sint_to_from_uint<T, R>::value &&
                     is_standard_type<OpenCLT>::value &&
                     is_standard_type<OpenCLR>::value,
                 R>
convertImpl(T Value) {
  return static_cast<R>(Value);
}

// sint to float
#define __SYCL_GENERATE_CONVERT_IMPL(SPIRVOp, DestType)                        \
  template <typename T, typename R, rounding_mode roundingMode,                \
            typename OpenCLT, typename OpenCLR>                                \
  std::enable_if_t<is_sint_to_float<T, R>::value &&                            \
                       (std::is_same_v<OpenCLR, DestType> ||                   \
                        (std::is_same_v<OpenCLR, _Float16> &&                  \
                         std::is_same_v<DestType, half>)),                     \
                   R>                                                          \
  convertImpl(T Value) {                                                       \
    OpenCLT OpValue = sycl::detail::convertDataToType<T, OpenCLT>(Value);      \
    return __spirv_Convert##SPIRVOp##_R##DestType(OpValue);                    \
  }

__SYCL_GENERATE_CONVERT_IMPL(SToF, half)
__SYCL_GENERATE_CONVERT_IMPL(SToF, float)
__SYCL_GENERATE_CONVERT_IMPL(SToF, double)

#undef __SYCL_GENERATE_CONVERT_IMPL

// uint to float
#define __SYCL_GENERATE_CONVERT_IMPL(SPIRVOp, DestType)                        \
  template <typename T, typename R, rounding_mode roundingMode,                \
            typename OpenCLT, typename OpenCLR>                                \
  std::enable_if_t<is_uint_to_float<T, R>::value &&                            \
                       (std::is_same_v<OpenCLR, DestType> ||                   \
                        (std::is_same_v<OpenCLR, _Float16> &&                  \
                         std::is_same_v<DestType, half>)),                     \
                   R>                                                          \
  convertImpl(T Value) {                                                       \
    OpenCLT OpValue = sycl::detail::convertDataToType<T, OpenCLT>(Value);      \
    return __spirv_Convert##SPIRVOp##_R##DestType(OpValue);                    \
  }

__SYCL_GENERATE_CONVERT_IMPL(UToF, half)
__SYCL_GENERATE_CONVERT_IMPL(UToF, float)
__SYCL_GENERATE_CONVERT_IMPL(UToF, double)

#undef __SYCL_GENERATE_CONVERT_IMPL

// float to float
#define __SYCL_GENERATE_CONVERT_IMPL(DestType, RoundingMode,                   \
                                     RoundingModeCondition)                    \
  template <typename T, typename R, rounding_mode roundingMode,                \
            typename OpenCLT, typename OpenCLR>                                \
  std::enable_if_t<is_float_to_float<T, R>::value &&                           \
                       !std::is_same_v<OpenCLT, OpenCLR> &&                    \
                       (std::is_same_v<OpenCLR, DestType> ||                   \
                        (std::is_same_v<OpenCLR, _Float16> &&                  \
                         std::is_same_v<DestType, half>)) &&                   \
                       RoundingModeCondition<roundingMode>::value,             \
                   R>                                                          \
  convertImpl(T Value) {                                                       \
    OpenCLT OpValue = sycl::detail::convertDataToType<T, OpenCLT>(Value);      \
    return __spirv_FConvert##_R##DestType##_##RoundingMode(OpValue);           \
  }

#define __SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(RoundingMode,           \
                                                       RoundingModeCondition)  \
  __SYCL_GENERATE_CONVERT_IMPL(double, RoundingMode, RoundingModeCondition)    \
  __SYCL_GENERATE_CONVERT_IMPL(float, RoundingMode, RoundingModeCondition)     \
  __SYCL_GENERATE_CONVERT_IMPL(half, RoundingMode, RoundingModeCondition)

__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rte, RteOrAutomatic)
__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rtz, Rtz)
__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rtp, Rtp)
__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rtn, Rtn)

#undef __SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE
#undef __SYCL_GENERATE_CONVERT_IMPL

// float to int
#define __SYCL_GENERATE_CONVERT_IMPL(SPIRVOp, DestType, RoundingMode,          \
                                     RoundingModeCondition)                    \
  template <typename T, typename R, rounding_mode roundingMode,                \
            typename OpenCLT, typename OpenCLR>                                \
  std::enable_if_t<is_float_to_int<T, R>::value &&                             \
                       (std::is_same_v<OpenCLR, opencl::cl_##DestType> ||      \
                        (std::is_same_v<OpenCLR, signed char> &&               \
                         std::is_same_v<DestType, char>)) &&                   \
                       RoundingModeCondition<roundingMode>::value,             \
                   R>                                                          \
  convertImpl(T Value) {                                                       \
    OpenCLT OpValue = sycl::detail::convertDataToType<T, OpenCLT>(Value);      \
    return __spirv_Convert##SPIRVOp##_R##DestType##_##RoundingMode(OpValue);   \
  }

#define __SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(RoundingMode,           \
                                                       RoundingModeCondition)  \
  __SYCL_GENERATE_CONVERT_IMPL(FToS, int, RoundingMode, RoundingModeCondition) \
  __SYCL_GENERATE_CONVERT_IMPL(FToS, char, RoundingMode,                       \
                               RoundingModeCondition)                          \
  __SYCL_GENERATE_CONVERT_IMPL(FToS, short, RoundingMode,                      \
                               RoundingModeCondition)                          \
  __SYCL_GENERATE_CONVERT_IMPL(FToS, long, RoundingMode,                       \
                               RoundingModeCondition)                          \
  __SYCL_GENERATE_CONVERT_IMPL(FToU, uint, RoundingMode,                       \
                               RoundingModeCondition)                          \
  __SYCL_GENERATE_CONVERT_IMPL(FToU, uchar, RoundingMode,                      \
                               RoundingModeCondition)                          \
  __SYCL_GENERATE_CONVERT_IMPL(FToU, ushort, RoundingMode,                     \
                               RoundingModeCondition)                          \
  __SYCL_GENERATE_CONVERT_IMPL(FToU, ulong, RoundingMode, RoundingModeCondition)

__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rte, RteOrAutomatic)
__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rtz, Rtz)
__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rtp, Rtp)
__SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE(rtn, Rtn)

#undef __SYCL_GENERATE_CONVERT_IMPL_FOR_ROUNDING_MODE
#undef __SYCL_GENERATE_CONVERT_IMPL

// Back up
template <typename T, typename R, rounding_mode roundingMode, typename OpenCLT,
          typename OpenCLR>
std::enable_if_t<
    ((!is_standard_type<T>::value && !is_standard_type<OpenCLT>::value) ||
     (!is_standard_type<R>::value && !is_standard_type<OpenCLR>::value)) &&
        !std::is_same_v<OpenCLT, OpenCLR>,
    R>
convertImpl(T Value) {
  return static_cast<R>(Value);
}

#endif // __SYCL_DEVICE_ONLY__
} // namespace detail
} // namespace _V1
} // namespace sycl
