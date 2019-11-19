// RUN: %clangxx -fsycl %s -o %t.out
//==------------ aliases.cpp - SYCL type aliases test ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <type_traits>

using std::is_same;
namespace s = cl::sycl;

// Test to verify requirements from 4.10.2.2 Aliases

#define ASSERT(ALIAS, TYPE, N)                                                 \
  static_assert(is_same<ALIAS##N, s::vec<TYPE, N>>::value, "");

// char, short, int, long, float, double, half
#define PRESENTED_AS_IS(N)                                                     \
  ASSERT(s::char, char, N)                                                     \
  ASSERT(s::short, short, N)                                                   \
  ASSERT(s::int, int, N)                                                       \
  ASSERT(s::long, long, N)                                                     \
  ASSERT(s::float, float, N)                                                   \
  ASSERT(s::double, double, N)                                                 \
  ASSERT(s::half, half, N);

// cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint, cl_long, cl_ulong,
// cl_float, cl_double and cl_half
#define OPENCL_INTEROPERABILITY(N)                                             \
  ASSERT(s::cl_char, s::cl_char, N)                                            \
  ASSERT(s::cl_uchar, s::cl_uchar, N)                                          \
  ASSERT(s::cl_short, s::cl_short, N)                                          \
  ASSERT(s::cl_ushort, s::cl_ushort, N)                                        \
  ASSERT(s::cl_int, s::cl_int, N)                                              \
  ASSERT(s::cl_uint, s::cl_uint, N)                                            \
  ASSERT(s::cl_long, s::cl_long, N)                                            \
  ASSERT(s::cl_ulong, s::cl_ulong, N)                                          \
  ASSERT(s::cl_float, s::cl_float, N)                                          \
  ASSERT(s::cl_double, s::cl_double, N)                                        \
  ASSERT(s::cl_half, s::cl_half, N)

// schar, uchar, ushort, uint, ulong, longlong and ulonglong
#define REPRESENTED_WITH_THE_SHORT_HAND(N)                                     \
  ASSERT(s::schar, signed char, N)                                             \
  ASSERT(s::uchar, unsigned char, N)                                           \
  ASSERT(s::ushort, unsigned short, N)                                         \
  ASSERT(s::uint, unsigned int, N)                                             \
  ASSERT(s::ulong, unsigned long, N)                                           \
  ASSERT(s::longlong, long long, N)                                            \
  ASSERT(s::ulonglong, unsigned long long, N)

#define CHECK_FOR_N(N)                                                         \
  PRESENTED_AS_IS(N)                                                           \
  OPENCL_INTEROPERABILITY(N)                                                   \
  REPRESENTED_WITH_THE_SHORT_HAND(N)

int main() {
  // For number of elements: 2, 3, 4, 8, 16.
  CHECK_FOR_N(2);
  CHECK_FOR_N(3);
  CHECK_FOR_N(4);
  CHECK_FOR_N(8);
  CHECK_FOR_N(16);
  return 0;
}
