//==----------- aliases.hpp --- SYCL type aliases --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/half_type.hpp>                 // for half

#include <cstdint> // for uint8_t, int16_t, int32_t

namespace sycl {
inline namespace _V1 {
template <typename T, int N> class __SYCL_EBO vec;
namespace detail::half_impl {
class half;
} // namespace detail::half_impl
} // namespace _V1
} // namespace sycl

#define __SYCL_MAKE_VECTOR_ALIAS(ALIAS, TYPE, N)                               \
  using ALIAS##N = sycl::vec<TYPE, N>;

#define __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(ALIAS, TYPE, N, MESSAGE)      \
  using ALIAS##N __SYCL2020_DEPRECATED(MESSAGE) = sycl::vec<TYPE, N>;

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES(N)                     \
  __SYCL_MAKE_VECTOR_ALIAS(char, char, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(short, short, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(int, int, N)                                        \
  __SYCL_MAKE_VECTOR_ALIAS(long, long, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(float, float, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(double, double, N)                                  \
  __SYCL_MAKE_VECTOR_ALIAS(half, half, N)

// There are no 'cl_*' vec aliases in SYCL 2020
#define __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                         \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_char, sycl::cl_char, N, "")      \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_uchar, sycl::cl_uchar, N, "")    \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_short, sycl::cl_short, N, "")    \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_ushort, sycl::cl_ushort, N, "")  \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_int, sycl::cl_int, N, "")        \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_uint, sycl::cl_uint, N, "")      \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_long, sycl::cl_long, N, "")      \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_ulong, sycl::cl_ulong, N, "")    \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_float, sycl::cl_float, N, "")    \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_double, sycl::cl_double, N, "")  \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(cl_half, sycl::cl_half, N, "")

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES(N)            \
  __SYCL_MAKE_VECTOR_ALIAS(schar, signed char, N)                              \
  __SYCL_MAKE_VECTOR_ALIAS(uchar, unsigned char, N)                            \
  __SYCL_MAKE_VECTOR_ALIAS(ushort, unsigned short, N)                          \
  __SYCL_MAKE_VECTOR_ALIAS(uint, unsigned int, N)                              \
  __SYCL_MAKE_VECTOR_ALIAS(ulong, unsigned long, N)                            \
  __SYCL_MAKE_VECTOR_ALIAS(longlong, long long, N)                             \
  __SYCL_MAKE_VECTOR_ALIAS(ulonglong, unsigned long long, N)

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(N)                        \
  __SYCL_MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES(N)                           \
  __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                               \
  __SYCL_MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES(N)

// FIXME: OpenCL vector aliases are not defined by SYCL 2020 spec and should be
//        removed from here. See intel/llvm#7888. They are deprecated for now.
// FIXME: schar, longlong and ulonglong aliases are not defined by SYCL 2020
//        spec, but they are preserved in SYCL 2020 mode, because SYCL-CTS is
//        still using them.
//        See KhronosGroup/SYCL-CTS#446 and KhronosGroup/SYCL-Docs#335
#define __SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(N)                   \
  __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                               \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(schar, std::int8_t, N, "")          \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(longlong, std::int64_t, N, "")      \
  __SYCL_2020_MAKE_DEPRECATED_VECTOR_ALIAS(ulonglong, std::uint64_t, N, "")    \
  __SYCL_MAKE_VECTOR_ALIAS(char, std::int8_t, N)                               \
  __SYCL_MAKE_VECTOR_ALIAS(uchar, std::uint8_t, N)                             \
  __SYCL_MAKE_VECTOR_ALIAS(short, std::int16_t, N)                             \
  __SYCL_MAKE_VECTOR_ALIAS(ushort, std::uint16_t, N)                           \
  __SYCL_MAKE_VECTOR_ALIAS(int, std::int32_t, N)                               \
  __SYCL_MAKE_VECTOR_ALIAS(uint, std::uint32_t, N)                             \
  __SYCL_MAKE_VECTOR_ALIAS(long, std::int64_t, N)                              \
  __SYCL_MAKE_VECTOR_ALIAS(ulong, std::uint64_t, N)                            \
  __SYCL_MAKE_VECTOR_ALIAS(float, float, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(double, double, N)                                  \
  __SYCL_MAKE_VECTOR_ALIAS(half, half, N)

namespace sycl {
inline namespace _V1 {
using byte __SYCL2020_DEPRECATED("use std::byte instead") = std::uint8_t;
using schar __SYCL2020_DEPRECATED("") = signed char;
using uchar __SYCL2020_DEPRECATED("") = unsigned char;
using ushort __SYCL2020_DEPRECATED("") = unsigned short;
using uint __SYCL2020_DEPRECATED("") = unsigned int;
using ulong __SYCL2020_DEPRECATED("") = unsigned long;
using longlong __SYCL2020_DEPRECATED("") = long long;
using ulonglong __SYCL2020_DEPRECATED("") = unsigned long long;
using half = sycl::detail::half_impl::half;

using cl_bool __SYCL2020_DEPRECATED("use sycl::opencl::cl_bool instead") = bool;
using cl_char
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_char instead") = std::int8_t;
using cl_uchar
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_uchar instead") = std::uint8_t;
using cl_short
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_short instead") = std::int16_t;
using cl_ushort __SYCL2020_DEPRECATED("use sycl::opencl::cl_ushort instead") =
    std::uint16_t;
using cl_int
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_int instead") = std::int32_t;
using cl_uint
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_uint instead") = std::uint32_t;
using cl_long
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_long instead") = std::int64_t;
using cl_ulong
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_ulong instead") = std::uint64_t;
using cl_half __SYCL2020_DEPRECATED("use sycl::opencl::cl_half instead") = half;
using cl_float
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_float instead") = float;
using cl_double
    __SYCL2020_DEPRECATED("use sycl::opencl::cl_double instead") = double;

namespace opencl {
// Strictly speaking, cl_* aliases should not be defined in opencl namespace in
// SYCL 1.2.1 mode, but we do so to simplify our implementation
using cl_bool = bool;
using cl_char = std::int8_t;
using cl_uchar = std::uint8_t;
using cl_short = std::int16_t;
using cl_ushort = std::uint16_t;
using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_long = std::int64_t;
using cl_ulong = std::uint64_t;
using cl_half = half;
using cl_float = float;
using cl_double = double;
} // namespace opencl

// Vector aliases are different between SYCL 1.2.1 and SYCL 2020
#if SYCL_LANGUAGE_VERSION >= 202001
__SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(2)
__SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(3)
__SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(4)
__SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(8)
__SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(16)
#else
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(2)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(3)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(4)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(8)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(16)
#endif
} // namespace _V1
} // namespace sycl

#undef __SYCL_MAKE_VECTOR_ALIAS
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH
#undef __SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH
