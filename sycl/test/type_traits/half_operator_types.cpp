// RUN: %clangxx -fsycl %s -o %t.out
//==-------------- type_traits.cpp - SYCL type_traits test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

template <typename T1, typename T_rtn> void check_half_math_operator_types() {
  static_assert(std::is_same<decltype(T1(1) + sycl::half(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(T1(1) - sycl::half(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(T1(1) * sycl::half(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(T1(1) / sycl::half(1)), T_rtn>::value);

  static_assert(std::is_same<decltype(sycl::half(1) + T1(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(sycl::half(1) - T1(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(sycl::half(1) * T1(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(sycl::half(1) / T1(1)), T_rtn>::value);
}

template <typename T1> void check_half_logical_operator_types() {
  static_assert(std::is_same<decltype(T1(1) == sycl::half(1)), bool>::value);
  static_assert(std::is_same<decltype(T1(1) != sycl::half(1)), bool>::value);
  static_assert(std::is_same<decltype(T1(1) > sycl::half(1)), bool>::value);
  static_assert(std::is_same<decltype(T1(1) < sycl::half(1)), bool>::value);
  static_assert(std::is_same<decltype(T1(1) <= sycl::half(1)), bool>::value);
  static_assert(std::is_same<decltype(T1(1) <= sycl::half(1)), bool>::value);

  static_assert(std::is_same<decltype(sycl::half(1) == T1(1)), bool>::value);
  static_assert(std::is_same<decltype(sycl::half(1) != T1(1)), bool>::value);
  static_assert(std::is_same<decltype(sycl::half(1) > T1(1)), bool>::value);
  static_assert(std::is_same<decltype(sycl::half(1) < T1(1)), bool>::value);
  static_assert(std::is_same<decltype(sycl::half(1) <= T1(1)), bool>::value);
  static_assert(std::is_same<decltype(sycl::half(1) <= T1(1)), bool>::value);
}

int main() {
  check_half_math_operator_types<sycl::half, sycl::half>();
  check_half_math_operator_types<double, double>();
  check_half_math_operator_types<float, float>();
  check_half_math_operator_types<int, sycl::half>();
  check_half_math_operator_types<long, sycl::half>();
  check_half_math_operator_types<long long, sycl::half>();

  check_half_logical_operator_types<sycl::half>();
  check_half_logical_operator_types<double>();
  check_half_logical_operator_types<float>();
  check_half_logical_operator_types<int>();
  check_half_logical_operator_types<long>();
  check_half_logical_operator_types<long long>();
}
