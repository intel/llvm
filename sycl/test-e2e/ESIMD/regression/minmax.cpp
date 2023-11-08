//==------------- minmax.cpp  - DPC++ ESIMD on-device test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that the call of esimd::min() and esimd::max() functions
// work correctly (even if windows.h or other OLD system header file is included
// and 'max' and 'min' are defined there as macros, and even more if those files
// are included without predefinition of NOMINMAX macro).

#if defined(_WIN32)
// NOMINMAX is intentionally NOT defined to cause potential problems with
// min/max functions.
// #define NOMINMAX
#include <windows.h>
#endif

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

int main() {
  constexpr unsigned VL = 32;

  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(Q);

  float *A = malloc_shared<float>(VL, Q);
  float *B = malloc_shared<float>(VL, Q);
  float *Res = malloc_shared<float>(2 * VL, Q);

  for (unsigned i = 0; i < VL; ++i) {
    A[i] = i;
    if (i % 3 == 0)
      B[i] = -i; // A > B
    else if (i % 3 == 1)
      B[i] = i; // A == B
    else if (i % 3 == 2)
      B[i] = i + 1000; // A >= B
    Res[i] = 0;
    Res[i + VL] = 0;
  }

  try {
    Q.single_task([=]() SYCL_ESIMD_KERNEL {
       using namespace sycl::ext::intel::esimd;
       simd<float, VL> VecA(A);
       simd<float, VL> VecB(B);
       auto MaxRes = (esimd::max)(VecA, VecB);
       auto MinRes = (esimd::min)(VecA, VecB);
       MinRes.copy_to(Res);
       MaxRes.copy_to(Res + VL);
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, Q);
    free(B, Q);
    free(Res, Q);
    return 1;
  }

  int NumErrors = 0;
  for (unsigned i = 0; i < VL; ++i) {
    float ExpectedMin = (std::min)(A[i], B[i]);
    float ExpectedMax = (std::max)(A[i], B[i]);
    if (Res[i] != ExpectedMin) {
      std::cout << "failed at min/index " << i << ", " << Res[i]
                << " != " << ExpectedMin << "\n";
      NumErrors++;
    }
    if (Res[i + VL] != ExpectedMax) {
      std::cout << "failed at max/index " << i << ", " << Res[i + VL]
                << " != " << ExpectedMax << "\n";
      NumErrors++;
    }
  }
  free(A, Q);
  free(B, Q);
  free(Res, Q);
  std::cout << ((NumErrors == 0) ? "Passed\n" : "FAILED\n");
  return NumErrors == 0 ? 0 : 1;
}
