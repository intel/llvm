// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==---------- recursion-in-constexpr.cpp - test recursion in constexpr ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/detail/core.hpp>


unsigned long long constexpr factorial(int n) {
  if (n == 0)
    return 1;
  return n * factorial(n - 1);
}

constexpr int X = 5;
constexpr int DataLen = 5;

template <int A> struct GetNTTP {
  static const int N = A;
};

int main() {
  sycl::queue q;
  int res[DataLen] = {0};
  {
  sycl::buffer<int> buf{res, {DataLen}};

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{buf, cgh};
    cgh.single_task([=] {
      constexpr int C = factorial(X);
      for (int i = 0; i < DataLen; ++i)
        acc[i] = C;
      acc[DataLen - 1] = GetNTTP<factorial(X)>::N;
    });
  });
  }

  for (int i = 0; i < DataLen; ++i) {
    if (res[i] != factorial(X)) {
      std::cout << "FAIL " << std::endl;
      return -1;
    }
  }

  return 0;
}
