//==----------- aliases.hpp --- SYCL type aliases --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>

#include <cstddef>
#include <cstdint>

__SYCL_OPEN_NS() {
template <typename T, int N> class vec;
namespace detail {
namespace half_impl {
class half;
} // namespace half_impl
} // namespace detail
} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()

using half __SYCL2020_DEPRECATED("use 'sycl::half' instead") =
    __sycl_ns::detail::half_impl::half;

#define __SYCL_MAKE_VECTOR_ALIAS(ALIAS, TYPE, N)                               \
  using ALIAS##N = __sycl_ns::vec<TYPE, N>;

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES(N)                     \
  __SYCL_MAKE_VECTOR_ALIAS(char, char, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(short, short, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(int, int, N)                                        \
  __SYCL_MAKE_VECTOR_ALIAS(long, long, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(float, float, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(double, double, N)                                  \
  __SYCL_MAKE_VECTOR_ALIAS(half, half, N)

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                         \
  __SYCL_MAKE_VECTOR_ALIAS(cl_char, __sycl_ns::cl_char, N)                     \
  __SYCL_MAKE_VECTOR_ALIAS(cl_uchar, __sycl_ns::cl_uchar, N)                   \
  __SYCL_MAKE_VECTOR_ALIAS(cl_short, __sycl_ns::cl_short, N)                   \
  __SYCL_MAKE_VECTOR_ALIAS(cl_ushort, __sycl_ns::cl_ushort, N)                 \
  __SYCL_MAKE_VECTOR_ALIAS(cl_int, __sycl_ns::cl_int, N)                       \
  __SYCL_MAKE_VECTOR_ALIAS(cl_uint, __sycl_ns::cl_uint, N)                     \
  __SYCL_MAKE_VECTOR_ALIAS(cl_long, __sycl_ns::cl_long, N)                     \
  __SYCL_MAKE_VECTOR_ALIAS(cl_ulong, __sycl_ns::cl_ulong, N)                   \
  __SYCL_MAKE_VECTOR_ALIAS(cl_float, __sycl_ns::cl_float, N)                   \
  __SYCL_MAKE_VECTOR_ALIAS(cl_double, __sycl_ns::cl_double, N)                 \
  __SYCL_MAKE_VECTOR_ALIAS(cl_half, __sycl_ns::cl_half, N)

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

__SYCL_OPEN_NS() {
using byte __SYCL2020_DEPRECATED("use std::byte instead") = std::uint8_t;
using schar = signed char;
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using longlong = long long;
using ulonglong = unsigned long long;
using half = __sycl_ns::detail::half_impl::half;
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

__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(2)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(3)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(4)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(8)
__SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(16)
} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()

#undef __SYCL_MAKE_VECTOR_ALIAS
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH
