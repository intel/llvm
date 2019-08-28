// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
//==------------ aliases.cpp - SYCL type aliases test ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <cassert>
#include <iostream>
#include <type_traits>

using namespace std;

using cl_schar = cl_char;
using cl_schar4 = cl_char4;

namespace s = cl::sycl;

#define CHECK_TYPE(TYPE)                                                       \
  static_assert(sizeof(cl_##TYPE) == sizeof(s::cl_##TYPE), "")

#define CHECK_SIZE(TYPE, SIZE) static_assert(sizeof(TYPE) == SIZE, "");

#define CHECK_SIZE_VEC_N(TYPE, N)                                              \
  static_assert(N * sizeof(TYPE) == sizeof(s::vec<TYPE, N>), "");

#define CHECK_SIZE_VEC_N3(TYPE)                                                \
  static_assert(sizeof(s::vec<TYPE, 3>) == sizeof(s::vec<TYPE, 4>), "");

#define CHECK_SIZE_VEC(TYPE)                                                   \
  CHECK_SIZE_VEC_N(TYPE, 2);                                                   \
  CHECK_SIZE_VEC_N3(TYPE);                                                     \
  CHECK_SIZE_VEC_N(TYPE, 4);                                                   \
  CHECK_SIZE_VEC_N(TYPE, 8);                                                   \
  CHECK_SIZE_VEC_N(TYPE, 16);

#define CHECK_SIZE_TYPE_I(TYPE, SIZE)                                          \
  CHECK_SIZE(TYPE, SIZE)                                                       \
  static_assert(std::is_signed<TYPE>::value, "");

#define CHECK_SIZE_TYPE_UI(TYPE, SIZE)                                         \
  CHECK_SIZE(TYPE, SIZE)                                                       \
  static_assert(std::is_unsigned<TYPE>::value, "");

#define CHECK_SIZE_TYPE_F(TYPE, SIZE)                                          \
  CHECK_SIZE(TYPE, SIZE)                                                       \
  static_assert(std::numeric_limits<TYPE>::is_iec559, "");

int main() {
  CHECK_TYPE(bool);
  CHECK_TYPE(char);
  CHECK_TYPE(schar);
  CHECK_TYPE(uchar);
  CHECK_TYPE(short);
  CHECK_TYPE(ushort);
  CHECK_TYPE(half);
  CHECK_TYPE(int);
  CHECK_TYPE(uint);
  CHECK_TYPE(long);
  CHECK_TYPE(ulong);
  CHECK_TYPE(float);
  CHECK_TYPE(double);
  CHECK_TYPE(char2);
  CHECK_TYPE(uchar3);
  CHECK_TYPE(short4);
  CHECK_TYPE(ushort8);
  CHECK_TYPE(half16);
  CHECK_TYPE(int2);
  CHECK_TYPE(uint3);
  CHECK_TYPE(long4);
  CHECK_TYPE(schar4);
  CHECK_TYPE(ulong8);
  CHECK_TYPE(float16);
  CHECK_TYPE(double2);

  // Table 4.93: Scalar data type aliases supported by SYCL
  CHECK_SIZE_TYPE_UI(s::byte, 1);

  CHECK_SIZE_TYPE_I(s::cl_char, 1);
  CHECK_SIZE_TYPE_I(s::cl_short, 2);
  CHECK_SIZE_TYPE_I(s::cl_int, 4);
  CHECK_SIZE_TYPE_I(s::cl_long, 8);

  CHECK_SIZE_TYPE_UI(s::cl_uchar, 1);
  CHECK_SIZE_TYPE_UI(s::cl_ushort, 2);
  CHECK_SIZE_TYPE_UI(s::cl_uint, 4);
  CHECK_SIZE_TYPE_UI(s::cl_ulong, 8);

  CHECK_SIZE_TYPE_F(s::cl_float, 4);
  CHECK_SIZE_TYPE_F(s::cl_double, 8);
  CHECK_SIZE(s::cl_half, 2);

  CHECK_SIZE_VEC(s::cl_char);
  CHECK_SIZE_VEC(s::cl_schar);
  CHECK_SIZE_VEC(s::cl_uchar);
  CHECK_SIZE_VEC(s::cl_short);
  CHECK_SIZE_VEC(s::cl_ushort);
  CHECK_SIZE_VEC(s::cl_half);
  CHECK_SIZE_VEC(s::cl_int);
  CHECK_SIZE_VEC(s::cl_uint);
  CHECK_SIZE_VEC(s::cl_long);
  CHECK_SIZE_VEC(s::cl_ulong);
  CHECK_SIZE_VEC(s::cl_float);
  CHECK_SIZE_VEC(s::cl_double);
}
