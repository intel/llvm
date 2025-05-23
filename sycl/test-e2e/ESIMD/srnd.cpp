//==---------------- srnd.cpp - DPC++ ESIMD srnd function test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "esimd_test_utils.hpp"

#include <cmath>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

bool test(queue &Q) {

  constexpr int N = 8;
  float *Input = malloc_shared<float>(N, Q);
  half *Output = malloc_shared<half>(N * 2, Q);

  for (int i = 0; i < N; ++i) {
    float value = esimd_test::getRandomValue<float>();
    while (sycl::fabs(value) > std::numeric_limits<half>::max() ||
           sycl::fabs(value) < std::numeric_limits<half>::min() ||
           std::isnan(value))
      value = esimd_test::getRandomValue<float>();

    Input[i] = value;
  }

  Q.single_task([=]() SYCL_ESIMD_KERNEL {
     simd<float, N> InputVector;
     InputVector.copy_from(Input);
     {
       simd<uint16_t, N> RandomVector(0);
       simd<half, N> OutputVector = srnd(InputVector, RandomVector);
       OutputVector.copy_to(Output);
     }
     {
       simd<uint16_t, N> RandomVector(0xFFFF);
       simd<half, N> OutputVector = srnd(InputVector, RandomVector);
       OutputVector.copy_to(Output + N);
     }
   }).wait();
  bool ReturnValue = true;
  for (int i = 0; i < N; ++i) {
    if (Output[i + N] == Output[i]) {
      ReturnValue = false;
      break;
    }
  }

  free(Input, Q);
  free(Output, Q);
  return ReturnValue;
}

// --- The entry point.

int main(void) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(Q);
  bool Pass = true;

  Pass &= test(Q);

  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
