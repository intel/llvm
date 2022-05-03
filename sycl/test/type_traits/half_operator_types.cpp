// RUN: %clangxx -fsycl %s -o %t.out
//==-------------- type_traits.cpp - SYCL type_traits test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

template <typename T1, typename T_rtn>
void check_half_math_operator_types(sycl::queue Queue) {

  // Test on host
  static_assert(std::is_same_v<decltype(T1(1) + sycl::half(1)), T_rtn>);
  static_assert(std::is_same_v<decltype(T1(1) - sycl::half(1)), T_rtn>);
  static_assert(std::is_same_v<decltype(T1(1) * sycl::half(1)), T_rtn>);
  static_assert(std::is_same_v<decltype(T1(1) / sycl::half(1)), T_rtn>);

  static_assert(std::is_same_v<decltype(sycl::half(1) + T1(1)), T_rtn>);
  static_assert(std::is_same_v<decltype(sycl::half(1) - T1(1)), T_rtn>);
  static_assert(std::is_same_v<decltype(sycl::half(1) * T1(1)), T_rtn>);
  static_assert(std::is_same_v<decltype(sycl::half(1) / T1(1)), T_rtn>);

  // Test on device
  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task([=] {
      static_assert(std::is_same_v<decltype(T1(1) + sycl::half(1)), T_rtn>);
      static_assert(std::is_same_v<decltype(T1(1) - sycl::half(1)), T_rtn>);
      static_assert(std::is_same_v<decltype(T1(1) * sycl::half(1)), T_rtn>);
      static_assert(std::is_same_v<decltype(T1(1) / sycl::half(1)), T_rtn>);

      static_assert(std::is_same_v<decltype(sycl::half(1) + T1(1)), T_rtn>);
      static_assert(std::is_same_v<decltype(sycl::half(1) - T1(1)), T_rtn>);
      static_assert(std::is_same_v<decltype(sycl::half(1) * T1(1)), T_rtn>);
      static_assert(std::is_same_v<decltype(sycl::half(1) / T1(1)), T_rtn>);
    });
  });
}

template <typename T1>
void check_half_logical_operator_types(sycl::queue Queue) {

  // Test on host
  static_assert(std::is_same_v<decltype(T1(1) == sycl::half(1)), bool>);
  static_assert(std::is_same_v<decltype(T1(1) != sycl::half(1)), bool>);
  static_assert(std::is_same_v<decltype(T1(1) > sycl::half(1)), bool>);
  static_assert(std::is_same_v<decltype(T1(1) < sycl::half(1)), bool>);
  static_assert(std::is_same_v<decltype(T1(1) <= sycl::half(1)), bool>);
  static_assert(std::is_same_v<decltype(T1(1) <= sycl::half(1)), bool>);

  static_assert(std::is_same_v<decltype(sycl::half(1) == T1(1)), bool>);
  static_assert(std::is_same_v<decltype(sycl::half(1) != T1(1)), bool>);
  static_assert(std::is_same_v<decltype(sycl::half(1) > T1(1)), bool>);
  static_assert(std::is_same_v<decltype(sycl::half(1) < T1(1)), bool>);
  static_assert(std::is_same_v<decltype(sycl::half(1) <= T1(1)), bool>);
  static_assert(std::is_same_v<decltype(sycl::half(1) <= T1(1)), bool>);

  // Test on device
  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task([=] {
      static_assert(std::is_same_v<decltype(T1(1) == sycl::half(1)), bool>);
      static_assert(std::is_same_v<decltype(T1(1) != sycl::half(1)), bool>);
      static_assert(std::is_same_v<decltype(T1(1) > sycl::half(1)), bool>);
      static_assert(std::is_same_v<decltype(T1(1) < sycl::half(1)), bool>);
      static_assert(std::is_same_v<decltype(T1(1) <= sycl::half(1)), bool>);
      static_assert(std::is_same_v<decltype(T1(1) <= sycl::half(1)), bool>);

      static_assert(std::is_same_v<decltype(sycl::half(1) == T1(1)), bool>);
      static_assert(std::is_same_v<decltype(sycl::half(1) != T1(1)), bool>);
      static_assert(std::is_same_v<decltype(sycl::half(1) > T1(1)), bool>);
      static_assert(std::is_same_v<decltype(sycl::half(1) < T1(1)), bool>);
      static_assert(std::is_same_v<decltype(sycl::half(1) <= T1(1)), bool>);
      static_assert(std::is_same_v<decltype(sycl::half(1) <= T1(1)), bool>);
    });
  });
}

int main() {

  sycl::queue Queue;

  check_half_math_operator_types<sycl::half, sycl::half>(Queue);
  check_half_math_operator_types<double, double>(Queue);
  check_half_math_operator_types<float, float>(Queue);
  check_half_math_operator_types<int, sycl::half>(Queue);
  check_half_math_operator_types<long, sycl::half>(Queue);
  check_half_math_operator_types<long long, sycl::half>(Queue);

  check_half_logical_operator_types<sycl::half>(Queue);
  check_half_logical_operator_types<double>(Queue);
  check_half_logical_operator_types<float>(Queue);
  check_half_logical_operator_types<int>(Queue);
  check_half_logical_operator_types<long>(Queue);
  check_half_logical_operator_types<long long>(Queue);
}
