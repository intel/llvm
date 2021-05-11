// RUN: %clangxx -fsycl %s -o %t.out
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

void check_vectors(sycl::int4 a, sycl::int4 b, sycl::int4 c, sycl::int4 gold) {
  sycl::int4 result = a * (sycl::int4)b.y() + c;
  assert((int)result.x() == (int)gold.x());
  assert((int)result.y() == (int)gold.y());
  assert((int)result.z() == (int)gold.z());
  assert((int)result.w() == (int)gold.w());
}

template <typename From, typename To> void check_convert() {
  sycl::vec<From, 4> vec{1, 2, 3, 4};
  sycl::vec<To, 4> result = vec.template convert<To>();
  assert((int)result.x() == (int)vec.x());
  assert((int)result.y() == (int)vec.y());
  assert((int)result.z() == (int)vec.z());
  assert((int)result.w() == (int)vec.w());
}

template <typename From, typename To> void check_signed_unsigned_convert_to() {
  check_convert<From, To>();
  check_convert<From, sycl::detail::make_unsigned_t<To>>();
  check_convert<sycl::detail::make_unsigned_t<From>, To>();
  check_convert<sycl::detail::make_unsigned_t<From>,
                sycl::detail::make_unsigned_t<To>>();
}

template <typename From> void check_convert_from() {
  check_signed_unsigned_convert_to<From, int8_t>();
  check_signed_unsigned_convert_to<From, int16_t>();
  check_signed_unsigned_convert_to<From, int32_t>();
  check_signed_unsigned_convert_to<From, int64_t>();
  check_signed_unsigned_convert_to<From, char>();
  check_signed_unsigned_convert_to<From, short>();
  check_signed_unsigned_convert_to<From, int>();
  check_signed_unsigned_convert_to<From, long>();
  check_signed_unsigned_convert_to<From, long long>();
  check_signed_unsigned_convert_to<From, half>();
  check_signed_unsigned_convert_to<From, float>();
  check_signed_unsigned_convert_to<From, double>();
}

int main() {
  sycl::int4 a = {1, 2, 3, 4};
  const sycl::int4 b = {10, 20, 30, 40};
  const sycl::int4 gold = {21, 42, 90, 120};
  const sycl::int2 a_xy = a.xy();
  check_vectors(a, b, {1, 2, 30, 40}, gold);
  check_vectors(a, b, {a.x(), a.y(), b.z(), b.w()}, gold);
  check_vectors(a, b, {a.x(), 2, b.z(), 40}, gold);
  check_vectors(a, b, {a.x(), 2, b.zw()}, gold);
  check_vectors(a, b, {a_xy, b.z(), 40}, gold);
  check_vectors(a, b, {a.xy(), b.zw()}, gold);

  // Constructing vector from a scalar
  sycl::vec<int, 1> vec_from_one_elem(1);

  // implicit conversion
  sycl::vec<unsigned char, 2> vec_2(1, 2);
  sycl::vec<unsigned char, 4> vec_4(0, vec_2, 3);

  assert(vec_4.get_count() == 4);
  assert(static_cast<unsigned char>(vec_4.x()) == static_cast<unsigned char>(0));
  assert(static_cast<unsigned char>(vec_4.y()) == static_cast<unsigned char>(1));
  assert(static_cast<unsigned char>(vec_4.z()) == static_cast<unsigned char>(2));
  assert(static_cast<unsigned char>(vec_4.w()) == static_cast<unsigned char>(3));

  // explicit conversion
  int64_t(vec_2.x());
  sycl::int4(vec_2.x());

  // Check broadcasting operator=
  sycl::vec<float, 4> b_vec(1.0);
  b_vec = 0.5;
  assert(static_cast<float>(b_vec.x()) == static_cast<float>(0.5));
  assert(static_cast<float>(b_vec.y()) == static_cast<float>(0.5));
  assert(static_cast<float>(b_vec.z()) == static_cast<float>(0.5));
  assert(static_cast<float>(b_vec.w()) == static_cast<float>(0.5));

  // Check that vector with 'unsigned long long' elements has enough bits to
  // store value.
  unsigned long long ull_ref = 1ull - 2ull;
  auto ull_vec = sycl::vec<unsigned long long, 1>(ull_ref);
  unsigned long long ull_val = ull_vec.template swizzle<sycl::elem::s0>();
  assert(ull_val == ull_ref);

  // Check that the function as() in swizzle vec class is working correctly
  sycl::vec<int8_t, 2> inputVec = sycl::vec<int8_t, 2>(0, 1);
  auto asVec = inputVec.template swizzle<sycl::elem::s0, sycl::elem::s1>()
                   .template as<sycl::vec<int16_t, 1>>();

  // Check that [u]long[n] type aliases match vec<[unsigned] long, n> types.
  assert((std::is_same<sycl::vec<long, 2>, sycl::long2>::value));
  assert((std::is_same<sycl::vec<long, 3>, sycl::long3>::value));
  assert((std::is_same<sycl::vec<long, 4>, sycl::long4>::value));
  assert((std::is_same<sycl::vec<long, 8>, sycl::long8>::value));
  assert((std::is_same<sycl::vec<long, 16>, sycl::long16>::value));
  assert((std::is_same<sycl::vec<unsigned long, 2>, sycl::ulong2>::value));
  assert((std::is_same<sycl::vec<unsigned long, 3>, sycl::ulong3>::value));
  assert((std::is_same<sycl::vec<unsigned long, 4>, sycl::ulong4>::value));
  assert((std::is_same<sycl::vec<unsigned long, 8>, sycl::ulong8>::value));
  assert((std::is_same<sycl::vec<unsigned long, 16>, sycl::ulong16>::value));

  // Check convert() from and to various types.
  check_convert_from<int8_t>();
  check_convert_from<int16_t>();
  check_convert_from<int32_t>();
  check_convert_from<int64_t>();
  check_convert_from<char>();
  check_convert_from<short>();
  check_convert_from<int>();
  check_convert_from<long>();
  check_convert_from<long long>();
  check_convert_from<half>();
  check_convert_from<float>();
  check_convert_from<double>();

  return 0;
}
