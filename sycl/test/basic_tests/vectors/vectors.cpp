// RUN: %clangxx -fsycl %s -o %t_default.out
// RUN: %t_default.out
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes %s -o %t_vec.out %}
// RUN: %if preview-breaking-changes-supported %{ %t_vec.out %}

//==--------------- vectors.cpp - SYCL vectors test ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cstddef>

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
  if constexpr (std::is_same_v<To, bool>) {
    assert((bool)result.x() == (bool)vec.x());
    assert((bool)result.y() == (bool)vec.y());
    assert((bool)result.z() == (bool)vec.z());
    assert((bool)result.w() == (bool)vec.w());
  } else {
    assert((int)result.x() == (int)vec.x());
    assert((int)result.y() == (int)vec.y());
    assert((int)result.z() == (int)vec.z());
    assert((int)result.w() == (int)vec.w());
  }
}

template <class T>
constexpr auto has_unsigned_v =
    std::is_integral_v<T> && !std::is_same_v<T, bool>;

template <typename From, typename To> void check_signed_unsigned_convert_to() {
  check_convert<From, To>();
  if constexpr (has_unsigned_v<To>)
    check_convert<From, sycl::detail::make_unsigned_t<To>>();
  if constexpr (has_unsigned_v<From>)
    check_convert<sycl::detail::make_unsigned_t<From>, To>();
  if constexpr (has_unsigned_v<To> && has_unsigned_v<From>)
    check_convert<sycl::detail::make_unsigned_t<From>,
                  sycl::detail::make_unsigned_t<To>>();
}

template <typename From> void check_convert_from() {
  check_signed_unsigned_convert_to<From, int8_t>();
  check_signed_unsigned_convert_to<From, int16_t>();
  check_signed_unsigned_convert_to<From, int32_t>();
  check_signed_unsigned_convert_to<From, int64_t>();
  check_signed_unsigned_convert_to<From, bool>();
  check_signed_unsigned_convert_to<From, char>();
  check_signed_unsigned_convert_to<From, signed char>();
  check_signed_unsigned_convert_to<From, short>();
  check_signed_unsigned_convert_to<From, int>();
  check_signed_unsigned_convert_to<From, long>();
  check_signed_unsigned_convert_to<From, long long>();
  check_signed_unsigned_convert_to<From, sycl::half>();
  check_signed_unsigned_convert_to<From, float>();
  check_signed_unsigned_convert_to<From, double>();
}

template <typename T, typename OpT> void check_ops(OpT op, T c1, T c2) {
  auto check = [&](sycl::vec<T, 2> vres) {
    assert(op(c1, c2) == vres[0]);
    assert(op(c1, c2) == vres[1]);
  };

  sycl::vec<T, 2> v1(c1);
  sycl::vec<T, 2> v2(c2);
  check(op(v1.template swizzle<0, 1>(), v2.template swizzle<0, 1>()));
  check(op(v1.template swizzle<0, 1>(), v2));
  check(op(v1.template swizzle<0, 1>(), c2));
  check(op(c1, v2.template swizzle<0, 1>()));
  check(op(c1, v2));
  check(op(v1, v2.template swizzle<0, 1>()));
  check(op(v1, v2));
  check(op(v1, c2));

  sycl::vec<T, 2> v3 = {c1, c2};
  sycl::vec<T, 2> v4 = op(v3, v3.template swizzle<1, 0>());
  assert(v4[0] == op(c1, c2) && v4[1] == op(c2, c1));
  sycl::vec<T, 2> v5 = op(v3.template swizzle<1, 1>(), v3);
  assert(v5[0] == op(c2, c1) && v5[1] == op(c2, c2));
  sycl::vec<T, 2> v6 =
      op(v3.template swizzle<1, 1>(), v3.template swizzle<0, 0>());
  assert(v6[0] == op(c2, c1) && v6[1] == op(c2, c1));
}

int main() {
  sycl::int4 a = {1, 2, 3, 4};
  const sycl::int4 b = {10, 20, 30, 40};
  const sycl::int4 gold = {21, 42, 90, 120};
  const sycl::int2 a_xy = a.lo();
  check_vectors(a, b, {1, 2, 30, 40}, gold);
  check_vectors(a, b, {a.x(), a.y(), b.z(), b.w()}, gold);
  check_vectors(a, b, {a.x(), 2, b.z(), 40}, gold);
  check_vectors(a, b, {a.x(), 2, b.hi()}, gold);
  check_vectors(a, b, {a.lo(), b.z(), 40}, gold);
  check_vectors(a, b, {a.lo(), b.hi()}, gold);

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
  b_vec.swizzle<0, 1, 2, 3>() = 0.6;
  assert(static_cast<float>(b_vec.x()) == static_cast<float>(0.6));
  assert(static_cast<float>(b_vec.y()) == static_cast<float>(0.6));
  assert(static_cast<float>(b_vec.z()) == static_cast<float>(0.6));
  assert(static_cast<float>(b_vec.w()) == static_cast<float>(0.6));

  // Check that vector with 'unsigned long long' elements has enough bits to
  // store value.
  unsigned long long ull_ref = 1ull - 2ull;
  auto ull_vec = sycl::vec<unsigned long long, 1>(ull_ref);
  unsigned long long ull_val = ull_vec.template swizzle<sycl::elem::s0>();
  assert(ull_val == ull_ref);

  // Check the swizzle vec class interface.
  static_assert(
      std::is_same_v<
          decltype(a.template swizzle<sycl::elem::s0>())::element_type, int>);
  const int &b_elem0_const = b.template swizzle<sycl::elem::s0>()[0];
  const int &a_elem0_const = a.template swizzle<sycl::elem::s0>()[0];
  int &a_elem0 = a.template swizzle<sycl::elem::s0>()[0];
  assert(b_elem0_const == 10);
  assert(a_elem0_const == 1);
  assert(a_elem0 == 1);

  // Check that the function as() in swizzle vec class is working correctly
  sycl::vec<int8_t, 2> inputVec = sycl::vec<int8_t, 2>(0, 1);
  auto asVec = inputVec.template swizzle<sycl::elem::s0, sycl::elem::s1>()
                   .template as<sycl::vec<int16_t, 1>>();
  auto test = inputVec.as<sycl::vec<bool, 2>>();
  assert(!test[0] && test[1]);
  sycl::vec<int8_t, 4> inputVec4 = sycl::vec<int8_t, 4>(0, 1, 1, 0);
  assert((!inputVec4.lo().as<sycl::vec<bool, 2>>()[0]));
  assert((inputVec4.lo().as<sycl::vec<bool, 2>>()[1]));

  // Check that [u]long[n] type aliases match vec<[u]int64_t, n> types.
  assert((std::is_same<sycl::vec<std::int64_t, 2>, sycl::long2>::value));
  assert((std::is_same<sycl::vec<std::int64_t, 3>, sycl::long3>::value));
  assert((std::is_same<sycl::vec<std::int64_t, 4>, sycl::long4>::value));
  assert((std::is_same<sycl::vec<std::int64_t, 8>, sycl::long8>::value));
  assert((std::is_same<sycl::vec<std::int64_t, 16>, sycl::long16>::value));
  assert((std::is_same<sycl::vec<std::uint64_t, 2>, sycl::ulong2>::value));
  assert((std::is_same<sycl::vec<std::uint64_t, 3>, sycl::ulong3>::value));
  assert((std::is_same<sycl::vec<std::uint64_t, 4>, sycl::ulong4>::value));
  assert((std::is_same<sycl::vec<std::uint64_t, 8>, sycl::ulong8>::value));
  assert((std::is_same<sycl::vec<std::uint64_t, 16>, sycl::ulong16>::value));

  // Check convert() from and to various types.
  check_convert_from<int8_t>();
  check_convert_from<int16_t>();
  check_convert_from<int32_t>();
  check_convert_from<int64_t>();
  check_convert_from<char>();
  check_convert_from<signed char>();
  check_convert_from<short>();
  check_convert_from<int>();
  check_convert_from<long>();
  check_convert_from<long long>();
  check_convert_from<sycl::half>();
  check_convert_from<float>();
  check_convert_from<double>();
  check_convert_from<bool>();

  check_ops<int>(std::modulus(), 6, 3);

  return 0;
}
