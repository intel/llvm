//==---------------- dpas_tf32.cpp  - DPC++ ESIMD on-device test ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: esimd_emulator

// The test verifies the low-level API for DPAS with 'tfloat32' types.
// It checks the versions of DPAS with and without the accumulator operand.

#include "../../esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

int main() {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  constexpr int REPEAT_COUNT = 8;
  constexpr int SYSTOLIC_DEPTH = 8;
  constexpr int EXECUTION_SIZE = 16;

  constexpr int M = REPEAT_COUNT;
  constexpr int N = EXECUTION_SIZE;
  constexpr int K = SYSTOLIC_DEPTH; // SYSTOLIC_DEPTH * OPS_PER_CHANNEL
  float *A = malloc_shared<float>(M * K, Q);
  float *B = malloc_shared<float>(K * N, Q);
  float *C = malloc_shared<float>(M * N, Q);
  float *D = malloc_shared<float>(M * N, Q);
  for (int I = 0; I < M * K; ++I)
    A[I] = I;
  for (int I = 0; I < K * N; ++I)
    B[I] = I;

  Q.single_task([=]() SYCL_ESIMD_KERNEL {
     simd<float, M * K> AVec(A);
     simd<float, K * N> BVec(B);
     auto AView = AVec.template bit_cast_view<uint>();
     auto BView = BVec.template bit_cast_view<uint>();
     // C(MxN) = A(MxK) * B(KxN)
     simd<float, M *N> CVec =
         dpas<argument_type::TF32, argument_type::TF32, SYSTOLIC_DEPTH,
              REPEAT_COUNT, float, uint, uint, M * N, K * N, M * K>(
             BView.read(), AView.read());
     CVec.copy_to(C);

     // D(MxN) = D(MxN) + A(MxK) * B(KxN);
     simd<float, M *N> DVec = 1.0;
     DVec = dpas<argument_type::TF32, argument_type::TF32, SYSTOLIC_DEPTH,
                 REPEAT_COUNT, float, uint, uint, M * N, K * N, M * K>(
         DVec, BView.read(), AView.read());
     DVec.copy_to(D);
   }).wait();

  unsigned ErrCnt = 0;
  for (unsigned I = 0; (I < M * N) && (ErrCnt < 10); ++I) {
    int m = I / N;
    int n = I % N;
    float RefResC = 0.0f;
    for (int k = 0; k < K; ++k)
      RefResC += float((m * K + k) * (k * N + n));
    if (std::abs(RefResC - C[I]) > 0.001) {
      std::cerr << "C[i] vs ref: " << C[I] << " : " << RefResC << std::endl;
      ErrCnt++;
    }
    float RefResD = RefResC + 1.0;
    if (std::abs(RefResD - D[I]) > 0.001) {
      std::cerr << "D[i] vs ref: " << D[I] << " : " << RefResD << std::endl;
      ErrCnt++;
    }
  }
  free(A, Q);
  free(B, Q);
  free(C, Q);
  free(D, Q);

  std::cout << (ErrCnt > 0 ? "FAILED\n" : "Passed\n");
  return ErrCnt > 0 ? 1 : 0;
}
