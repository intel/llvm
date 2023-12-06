// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==- bfloat16_vector_plus_scalar.cpp.cpp - Test for bfloat16 operators -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T> ESIMD_NOINLINE bool test(queue Q, int Case) {
  std::cout << "Testing case=" << Case << std::endl;

  constexpr int N = 8;
  T *Mem = malloc_shared<T>(N, Q);
  T TOne = static_cast<T>(1);
  T TTen = static_cast<T>(10);

  Q.single_task([=]() SYCL_ESIMD_KERNEL {
     simd<T, N> Vec(TOne);
     Vec = Vec + TTen;
     Vec.copy_to(Mem);
   }).wait();

  for (int i = 0; i < N; ++i) {
    if (Mem[i] != TOne + TTen) {
      return false;
    }
  }
  return true;
}

int main() {
  queue Q;
  std::cout << "Running on " << Q.get_device().get_info<info::device::name>()
            << "\n";

  bool Passed = true;
  Passed &= test<int>(Q, 1);
  Passed &= test<float>(Q, 2);
  Passed &= test<sycl::half>(Q, 3);
  Passed &= test<sycl::ext::oneapi::bfloat16>(Q, 4);
  Passed &= test<sycl::ext::intel::experimental::esimd::tfloat32>(Q, 5);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}