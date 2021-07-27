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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <typename T, int N> class vec;
namespace detail {
namespace half_impl {
class half;
} // namespace half_impl
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

// FIXME: line below exports 'half' into global namespace, which seems incorrect
// However, SYCL 1.2.1 spec considers 'half' to be a fundamental C++ data type
// which doesn't exist within the 'cl::sycl' namespace.
// Related spec issue: KhronosGroup/SYCL-Docs#40
using half = cl::sycl::detail::half_impl::half;

#define __SYCL_MAKE_VECTOR_ALIAS(ALIAS, TYPE, N)                               \
  using ALIAS##N = cl::sycl::vec<TYPE, N>;

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES(N)                     \
  __SYCL_MAKE_VECTOR_ALIAS(char, char, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(short, short, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(int, int, N)                                        \
  __SYCL_MAKE_VECTOR_ALIAS(long, long, N)                                      \
  __SYCL_MAKE_VECTOR_ALIAS(float, float, N)                                    \
  __SYCL_MAKE_VECTOR_ALIAS(double, double, N)                                  \
  __SYCL_MAKE_VECTOR_ALIAS(half, half, N)

#define __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                         \
  __SYCL_MAKE_VECTOR_ALIAS(cl_char, cl::sycl::cl_char, N)                      \
  __SYCL_MAKE_VECTOR_ALIAS(cl_uchar, cl::sycl::cl_uchar, N)                    \
  __SYCL_MAKE_VECTOR_ALIAS(cl_short, cl::sycl::cl_short, N)                    \
  __SYCL_MAKE_VECTOR_ALIAS(cl_ushort, cl::sycl::cl_ushort, N)                  \
  __SYCL_MAKE_VECTOR_ALIAS(cl_int, cl::sycl::cl_int, N)                        \
  __SYCL_MAKE_VECTOR_ALIAS(cl_uint, cl::sycl::cl_uint, N)                      \
  __SYCL_MAKE_VECTOR_ALIAS(cl_long, cl::sycl::cl_long, N)                      \
  __SYCL_MAKE_VECTOR_ALIAS(cl_ulong, cl::sycl::cl_ulong, N)                    \
  __SYCL_MAKE_VECTOR_ALIAS(cl_float, cl::sycl::cl_float, N)                    \
  __SYCL_MAKE_VECTOR_ALIAS(cl_double, cl::sycl::cl_double, N)                  \
  __SYCL_MAKE_VECTOR_ALIAS(cl_half, cl::sycl::cl_half, N)

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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
using byte __SYCL2020_DEPRECATED("use std::byte instead") = std::uint8_t;
using schar = signed char;
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using longlong = long long;
using ulonglong = unsigned long long;
using half = cl::sycl::detail::half_impl::half;
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
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#undef __SYCL_MAKE_VECTOR_ALIAS
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES
#undef __SYCL_MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH
