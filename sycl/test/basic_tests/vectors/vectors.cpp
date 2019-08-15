// RUN: %clangxx %s -o %t.out -lOpenCL -lsycl
// RUN: %t.out
//==--------------- vectors.cpp - SYCL vectors test ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>
using namespace cl::sycl;

void check_vectors(int4 a, int4 b, int4 c, int4 gold) {
  int4 result = a * (int4)b.y() + c;
  assert((int)result.x() == (int)gold.x());
  assert((int)result.y() == (int)gold.y());
  assert((int)result.w() == (int)gold.w());
  assert((int)result.z() == (int)gold.z());
}

int main() {
  int4 a = {1, 2, 3, 4};
  const int4 b = {10, 20, 30, 40};
  const int4 gold = {21, 42, 90, 120};
  const int2 a_xy = a.xy();
  check_vectors(a, b, {1, 2, 30, 40}, gold);
  check_vectors(a, b, {a.x(), a.y(), b.z(), b.w()}, gold);
  check_vectors(a, b, {a.x(), 2, b.z(), 40}, gold);
  check_vectors(a, b, {a.x(), 2, b.zw()}, gold);
  check_vectors(a, b, {a_xy, b.z(), 40}, gold);
  check_vectors(a, b, {a.xy(), b.zw()}, gold);

  // Constructing vector from a scalar
  cl::sycl::vec<int, 1> vec_from_one_elem(1);

  // implicit conversion
  cl::sycl::vec<unsigned char, 2> vec_2(1, 2);
  cl::sycl::vec<unsigned char, 4> vec_4(0, vec_2, 3);

  assert(vec_4.get_count() == 4);
  assert(static_cast<unsigned char>(vec_4.x()) == static_cast<unsigned char>(0));
  assert(static_cast<unsigned char>(vec_4.y()) == static_cast<unsigned char>(1));
  assert(static_cast<unsigned char>(vec_4.z()) == static_cast<unsigned char>(2));
  assert(static_cast<unsigned char>(vec_4.w()) == static_cast<unsigned char>(3));

  // explicit conversion
  int64_t(vec_2.x());
  cl::sycl::int4(vec_2.x());

  // Check broadcasting operator=
  cl::sycl::vec<float, 4> b_vec(1.0);
  b_vec = 0.5;
  assert(static_cast<float>(b_vec.x()) == static_cast<float>(0.5));
  assert(static_cast<float>(b_vec.y()) == static_cast<float>(0.5));
  assert(static_cast<float>(b_vec.z()) == static_cast<float>(0.5));
  assert(static_cast<float>(b_vec.w()) == static_cast<float>(0.5));

  // Check that vector with 'unsigned long long' elements has enough bits to
  // store value.
  unsigned long long ull_ref = 1ull - 2ull;
  auto ull_vec = cl::sycl::vec<unsigned long long, 1>(ull_ref);
  unsigned long long ull_val = ull_vec.template swizzle<cl::sycl::elem::s0>();
  assert(ull_val == ull_ref);

  // Check that [u]long[n] type aliases match vec<[unsigned] long, n> types.
  assert((std::is_same<cl::sycl::vec<long, 2>, cl::sycl::long2>::value));
  assert((std::is_same<cl::sycl::vec<long, 3>, cl::sycl::long3>::value));
  assert((std::is_same<cl::sycl::vec<long, 4>, cl::sycl::long4>::value));
  assert((std::is_same<cl::sycl::vec<long, 8>, cl::sycl::long8>::value));
  assert((std::is_same<cl::sycl::vec<long, 16>, cl::sycl::long16>::value));
  assert((std::is_same<cl::sycl::vec<unsigned long, 2>, cl::sycl::ulong2>::value));
  assert((std::is_same<cl::sycl::vec<unsigned long, 3>, cl::sycl::ulong3>::value));
  assert((std::is_same<cl::sycl::vec<unsigned long, 4>, cl::sycl::ulong4>::value));
  assert((std::is_same<cl::sycl::vec<unsigned long, 8>, cl::sycl::ulong8>::value));
  assert((std::is_same<cl::sycl::vec<unsigned long, 16>, cl::sycl::ulong16>::value));

  return 0;
}
