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

namespace cl {
namespace sycl {
template <typename T, int N> class vec;
namespace detail {
namespace half_impl {
class half;
} // namespace half_impl
} // namespace detail
} // namespace sycl
} // namespace cl

#ifdef __SYCL_DEVICE_ONLY__
using half = _Float16;
#else
using half = cl::sycl::detail::half_impl::half;
#endif

#define MAKE_VECTOR_ALIAS(ALIAS, TYPE, N)                                      \
  using ALIAS##N = cl::sycl::vec<TYPE, N>;

#define MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES(N)                            \
  MAKE_VECTOR_ALIAS(char, char, N)                                             \
  MAKE_VECTOR_ALIAS(short, short, N)                                           \
  MAKE_VECTOR_ALIAS(int, int, N)                                               \
  MAKE_VECTOR_ALIAS(long, long, N)                                             \
  MAKE_VECTOR_ALIAS(float, float, N)                                           \
  MAKE_VECTOR_ALIAS(double, double, N)                                         \
  MAKE_VECTOR_ALIAS(half, half, N)

#define MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                                \
  MAKE_VECTOR_ALIAS(cl_char, cl::sycl::cl_char, N)                             \
  MAKE_VECTOR_ALIAS(cl_uchar, cl::sycl::cl_uchar, N)                           \
  MAKE_VECTOR_ALIAS(cl_short, cl::sycl::cl_short, N)                           \
  MAKE_VECTOR_ALIAS(cl_ushort, cl::sycl::cl_ushort, N)                         \
  MAKE_VECTOR_ALIAS(cl_int, cl::sycl::cl_int, N)                               \
  MAKE_VECTOR_ALIAS(cl_uint, cl::sycl::cl_uint, N)                             \
  MAKE_VECTOR_ALIAS(cl_long, cl::sycl::cl_long, N)                             \
  MAKE_VECTOR_ALIAS(cl_ulong, cl::sycl::cl_ulong, N)                           \
  MAKE_VECTOR_ALIAS(cl_float, cl::sycl::cl_float, N)                           \
  MAKE_VECTOR_ALIAS(cl_double, cl::sycl::cl_double, N)                         \
  MAKE_VECTOR_ALIAS(cl_half, cl::sycl::cl_half, N)

#define MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES(N)                   \
  MAKE_VECTOR_ALIAS(schar, signed char, N)                                     \
  MAKE_VECTOR_ALIAS(uchar, unsigned char, N)                                   \
  MAKE_VECTOR_ALIAS(ushort, unsigned short, N)                                 \
  MAKE_VECTOR_ALIAS(uint, unsigned int, N)                                     \
  MAKE_VECTOR_ALIAS(ulong, unsigned long, N)                                   \
  MAKE_VECTOR_ALIAS(longlong, long long, N)                                    \
  MAKE_VECTOR_ALIAS(ulonglong, unsigned long long, N)

#define MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(N)                               \
  MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES(N)                                  \
  MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES(N)                                      \
  MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES(N)

namespace cl {
namespace sycl {
using byte = std::uint8_t;
using schar = signed char;
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using longlong = long long;
using ulonglong = unsigned long long;
// TODO cl::sycl::half is not in SYCL specification, but is used by Khronos CTS.
using half = half;
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

MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(2)
MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(3)
MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(4)
MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(8)
MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH(16)
} // namespace sycl
} // namespace cl

#undef MAKE_VECTOR_ALIAS
#undef MAKE_VECTOR_ALIASES_FOR_ARITHMETIC_TYPES
#undef MAKE_VECTOR_ALIASES_FOR_OPENCL_TYPES
#undef MAKE_VECTOR_ALIASES_FOR_SIGNED_AND_UNSIGNED_TYPES
#undef MAKE_VECTOR_ALIASES_FOR_VECTOR_LENGTH
