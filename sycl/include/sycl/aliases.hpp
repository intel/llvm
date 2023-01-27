//==----------- aliases.hpp --- SYCL type aliases --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cl.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/defines_elementary.hpp>

#include <cstddef>
#include <cstdint>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
template <typename T, int N> class vec;
namespace detail::half_impl {
class half;
} // namespace detail::half_impl
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#define __SYCL_MAKE_VECTOR_ALIAS(ALIAS, TYPE, N)                               \
  using ALIAS##N = sycl::vec<TYPE, N>;

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(N)                        \
  __SYCL_MAKE_VECTOR_ALIAS(char, char, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(short, short, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(int, int, N)                                        \
  __SYCL_MAKE_VECTOR_ALIAS(long, long, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(float, float, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(double, double, N)                                  \
  __SYCL_MAKE_VECTOR_ALIAS(half, half, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(schar, signed char, N)                              \
  __SYCL_MAKE_VECTOR_ALIAS(uchar, unsigned char, N)                            \
  __SYCL_MAKE_VECTOR_ALIAS(ushort, unsigned short, N)                          \
  __SYCL_MAKE_VECTOR_ALIAS(uint, unsigned int, N)                              \
  __SYCL_MAKE_VECTOR_ALIAS(ulong, unsigned long, N)                            \
  __SYCL_MAKE_VECTOR_ALIAS(longlong, long long, N)                             \
  __SYCL_MAKE_VECTOR_ALIAS(ulonglong, unsigned long long, N)

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                         \
  __SYCL_MAKE_VECTOR_ALIAS(cl_char, sycl::cl_char, N)                          \
  __SYCL_MAKE_VECTOR_ALIAS(cl_uchar, sycl::cl_uchar, N)                        \
  __SYCL_MAKE_VECTOR_ALIAS(cl_short, sycl::cl_short, N)                        \
  __SYCL_MAKE_VECTOR_ALIAS(cl_ushort, sycl::cl_ushort, N)                      \
  __SYCL_MAKE_VECTOR_ALIAS(cl_int, sycl::cl_int, N)                            \
  __SYCL_MAKE_VECTOR_ALIAS(cl_uint, sycl::cl_uint, N)                          \
  __SYCL_MAKE_VECTOR_ALIAS(cl_long, sycl::cl_long, N)                          \
  __SYCL_MAKE_VECTOR_ALIAS(cl_ulong, sycl::cl_ulong, N)                        \
  __SYCL_MAKE_VECTOR_ALIAS(cl_float, sycl::cl_float, N)                        \
  __SYCL_MAKE_VECTOR_ALIAS(cl_double, sycl::cl_double, N)                      \
  __SYCL_MAKE_VECTOR_ALIAS(cl_half, sycl::cl_half, N)


#define __SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(N)                   \
// FIXME: schar, longlong and ulonglong aliases are not defined by SYCL 2020
//        spec, but they are preserved in SYCL 2020 mode, because SYCL-CTS is
//        still using them.
//        See KhronosGroup/SYCL-CTS#446 and KhronosGroup/SYCL-Docs#335
#define __SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(N)                   \
  __SYCL_MAKE_VECTOR_ALIAS(schar, std::int8_t, N)                              \
  __SYCL_MAKE_VECTOR_ALIAS(longlong, std::int64_t, N)                          \
  __SYCL_MAKE_VECTOR_ALIAS(ulonglong, std::uint64_t, N)                        \
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
__SYCL_INLINE_VER_NAMESPACE(_V1) {
using byte __SYCL2020_DEPRECATED("use std::byte instead") = std::uint8_t;
using half = sycl::detail::half_impl::half;

namespace opencl {
// Strictly speaking, those alases do not exist in this namespace in SYCL 1.2.1,
// but we still provide them for the sake of implmentation simplicity
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

#if SYCL_LANGUAGE_VERSION < 202001
// These aliases in sycl namespace are provided with compatibility with
// SYCL 1.2.1

using schar = signed char;
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using longlong = long long;
using ulonglong = unsigned long long;

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
#endif

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
#if !defined(SYCL_LANGUAGE_VERSION) || SYCL_LANGUAGE_VERSION >= 202001
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

__SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(2)
__SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(3)
__SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(4)
__SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(8)
__SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(16)
#endif

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __SYCL_MAKE_VECTOR_ALIAS
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH
#undef __SYCL_2020_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH
