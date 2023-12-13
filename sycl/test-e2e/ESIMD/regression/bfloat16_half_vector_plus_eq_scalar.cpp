// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//= bfloat16_half_vector_plus_eq_scalar.cpp - Test for bfloat16 operators =//
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

  bool ReturnValue = true;
  for (int i = 0; i < N; ++i) {
    if (Mem[i] != TOne + TTen) {
      ReturnValue = false;
      break;
    }
    if (Mem[i + N] != TOne - TTen) {
      ReturnValue = false;
      break;
    }
    if (Mem[i + 2 * N] != TOne * TTen) {
      ReturnValue = false;
      break;
    }
    if (!((Mem[i + 3 * N] == (TOne / TTen)) ||
          (std::abs((double)(Mem[i + 3 * N] - (TOne / TTen)) /
                    (double)(TOne / TTen)) <= 0.001))) {
      ReturnValue = false;
      break;
    }
  }

  free(Mem, Q);
  return ReturnValue;
}

int main() {
  queue Q;
  esimd_test::printTestLabel(Q);

  bool SupportsHalf = Q.get_device().has(aspect::fp16);

  bool Passed = true;
  Passed &= test<int>(Q);
  Passed &= test<float>(Q);
  if (SupportsHalf) {
    Passed &= test<sycl::half>(Q);
  }

#ifdef USE_BF16
  Passed &= test<sycl::ext::oneapi::bfloat16>(Q);
#endif
#ifdef USE_TF32
  Passed &= test<sycl::ext::intel::experimental::esimd::tfloat32>(Q);
#endif
  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
