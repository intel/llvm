// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//==- bfloat16_vector_plus_eq_scalar.cpp.cpp - Test for bfloat16 operators -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../esimd_test_utils.hpp"
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T> ESIMD_NOINLINE bool test(queue Q) {
  std::cout << "Testing T=" << esimd_test::type_name<T>() << "...\n";

  constexpr int N = 8;

  constexpr int NumOps = 4;
  constexpr int CSize = NumOps * N;

  T *Mem = malloc_shared<T>(CSize, Q);
  T TOne = static_cast<T>(1);
  T TTen = static_cast<T>(10);

  Q.single_task([=]() SYCL_ESIMD_KERNEL {
     {
       simd<T, N> Vec(TOne);
       Vec += TTen;
       Vec.copy_to(Mem);
     }
     {
       simd<T, N> Vec(TOne);
       Vec -= TTen;
       Vec.copy_to(Mem + N);
     }
     {
       simd<T, N> Vec(TOne);
       Vec *= TTen;
       Vec.copy_to(Mem + 2 * N);
     }
     {
       simd<T, N> Vec(TOne);
       Vec /= TTen;
       Vec.copy_to(Mem + 3 * N);
     }
   }).wait();

  bool returnValue = true;
  for (int i = 0; i < N; ++i) {
    if (Mem[i] != TOne + TTen) {
      returnValue = false;
      break;
    }
    if (Mem[i + N] != TOne - TTen) {
      returnValue = false;
      break;
    }
    if (Mem[i + 2 * N] != TOne * TTen) {
      returnValue = false;
      break;
    }
    if (!((Mem[i + 3 * N] == (TOne / TTen)) ||
          (std::abs((double)(Mem[i + 3 * N] - (TOne / TTen)) /
                    (double)(TOne / TTen)) <= 0.001))) {
      returnValue = false;
      break;
    }
  }

  free(Mem, Q);
  return returnValue;
}

int main() {
  queue Q;
  std::cout << "Running on " << Q.get_device().get_info<info::device::name>()
            << "\n";

  bool Passed = true;
  Passed &= test<int>(Q);
  Passed &= test<float>(Q);
  Passed &= test<sycl::half>(Q);
  // TODO: Reenable once the issue with bfloat16 is resolved
  // Passed &= test<sycl::ext::oneapi::bfloat16>(Q);
  Passed &= test<sycl::ext::intel::experimental::esimd::tfloat32>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}