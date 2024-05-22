// RUN: %clangxx -fsycl -fsyntax-only %s
//==-------------- type_traits.cpp - SYCL type_traits test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
using namespace std;

template <typename T1, typename T_rtn> void math_operator_helper() {
  static_assert(
      is_same_v<decltype(declval<T1>() + declval<sycl::half>()), T_rtn>);
  static_assert(
      is_same_v<decltype(declval<T1>() - declval<sycl::half>()), T_rtn>);
  static_assert(
      is_same_v<decltype(declval<T1>() * declval<sycl::half>()), T_rtn>);
  static_assert(
      is_same_v<decltype(declval<T1>() / declval<sycl::half>()), T_rtn>);

  static_assert(
      is_same_v<decltype(declval<sycl::half>() + declval<T1>()), T_rtn>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() - declval<T1>()), T_rtn>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() * declval<T1>()), T_rtn>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() / declval<T1>()), T_rtn>);
}

template <typename T1> void logical_operator_helper() {
  static_assert(
      is_same_v<decltype(declval<T1>() == declval<sycl::half>()), bool>);
  static_assert(
      is_same_v<decltype(declval<T1>() != declval<sycl::half>()), bool>);
  static_assert(
      is_same_v<decltype(declval<T1>() > declval<sycl::half>()), bool>);
  static_assert(
      is_same_v<decltype(declval<T1>() < declval<sycl::half>()), bool>);
  static_assert(
      is_same_v<decltype(declval<T1>() <= declval<sycl::half>()), bool>);
  static_assert(
      is_same_v<decltype(declval<T1>() >= declval<sycl::half>()), bool>);

  static_assert(
      is_same_v<decltype(declval<sycl::half>() == declval<T1>()), bool>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() != declval<T1>()), bool>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() > declval<T1>()), bool>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() < declval<T1>()), bool>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() <= declval<T1>()), bool>);
  static_assert(
      is_same_v<decltype(declval<sycl::half>() >= declval<T1>()), bool>);
}

template <typename T1, typename T_rtn>
void check_half_math_operator_types(sycl::queue &Queue) {

  // Test on host
  math_operator_helper<T1, T_rtn>();

  // Test on device
  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task([=] { math_operator_helper<T1, T_rtn>(); });
  });
}

template <typename T1>
void check_half_logical_operator_types(sycl::queue &Queue) {

  // Test on host
  logical_operator_helper<T1>();

  // Test on device
  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task([=] { logical_operator_helper<T1>(); });
  });
}

template <typename T1>
void check_half_stream_operator_type(sycl::queue &Queue) {

  // Host only stream test
  std::istringstream iss;
  std::ostringstream oss;
  sycl::half val;
  static_assert(is_same_v<decltype(iss >> val), std::istream &>);
  static_assert(is_same_v<decltype(oss << val), std::ostream &>);
}

int main() {

  sycl::queue Queue;

  check_half_math_operator_types<sycl::half, sycl::half>(Queue);
  check_half_math_operator_types<double, double>(Queue);
  check_half_math_operator_types<float, float>(Queue);
  check_half_math_operator_types<int, sycl::half>(Queue);
  check_half_math_operator_types<long, sycl::half>(Queue);
  check_half_math_operator_types<long long, sycl::half>(Queue);
  check_half_math_operator_types<sycl::ext::oneapi::bfloat16,
                                 sycl::ext::oneapi::bfloat16>(Queue);

  check_half_logical_operator_types<sycl::half>(Queue);
  check_half_logical_operator_types<double>(Queue);
  check_half_logical_operator_types<float>(Queue);
  check_half_logical_operator_types<int>(Queue);
  check_half_logical_operator_types<long>(Queue);
  check_half_logical_operator_types<long long>(Queue);
  check_half_logical_operator_types<sycl::ext::oneapi::bfloat16>(Queue);
}
