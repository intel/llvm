// RUN: %clangxx -fsycl %s -fsyntax-only -o %t.out
//==-------------- type_traits.cpp - SYCL type_traits test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/half_type.hpp>

#include <iostream>
#include <type_traits>

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

void check_half_stream_operator_type() {
  std::istringstream iss;
  std::ostringstream oss;
  sycl::half val;
  static_assert(is_same_v<decltype(iss >> val), std::istream &>);
  static_assert(is_same_v<decltype(oss << val), std::ostream &>);
}

int main() {
  math_operator_helper<sycl::half, sycl::half>();
  math_operator_helper<double, double>();
  math_operator_helper<float, float>();
  math_operator_helper<int, sycl::half>();
  math_operator_helper<long, sycl::half>();
  math_operator_helper<long long, sycl::half>();

  logical_operator_helper<sycl::half>();
  logical_operator_helper<double>();
  logical_operator_helper<float>();
  logical_operator_helper<int>();
  logical_operator_helper<long>();
  logical_operator_helper<long long>();

  check_half_stream_operator_type();
}
