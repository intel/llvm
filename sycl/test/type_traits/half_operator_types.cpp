// RUN: %clangxx -fsycl %s -o %t.out
//==-------------- type_traits.cpp - SYCL type_traits test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

template <typename T1, typename T2, typename T_rtn>
void check_half_operator_types() {
  static_assert(std::is_same<decltype(T1(1) + T2(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(T1(1) - T2(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(T1(1) * T2(1)), T_rtn>::value);
  static_assert(std::is_same<decltype(T1(1) / T2(1)), T_rtn>::value);
}

int main() {
  check_half_operator_types<sycl::half, sycl::half, sycl::half>();
  check_half_operator_types<double, sycl::half, double>();
  check_half_operator_types<sycl::half, double, double>();
  check_half_operator_types<float, sycl::half, float>();
  check_half_operator_types<sycl::half, float, float>();

  check_half_operator_types<int, sycl::half, sycl::half>();
  check_half_operator_types<sycl::half, int, sycl::half>();
  check_half_operator_types<long, sycl::half, sycl::half>();
  check_half_operator_types<sycl::half, long, sycl::half>();
  check_half_operator_types<long long, sycl::half, sycl::half>();
  check_half_operator_types<sycl::half, long long, sycl::half>();
}
