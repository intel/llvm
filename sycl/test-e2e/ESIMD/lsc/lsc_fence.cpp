//==------------ lsc_fence.cpp - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the intrinsic lsc_fence on PVC.
// It is based on https://en.wikipedia.org/wiki/Memory_barrier#Example

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

// Returns 'true' if the test passes.
bool testGlobal(queue &Q, int OddOrEven /*0 - Even, 1 - Odd*/) {
  constexpr size_t LocalSize = 16;
  constexpr unsigned int SIMDSize = 8;
  constexpr size_t Size = LocalSize * SIMDSize * 64;

  int NumErrors = 0;
  std::cout << "Running testGlobal() with OddOrEven = " << OddOrEven << "...";

  int *A = aligned_alloc_shared<int>(128, Size, Q);
  int *B = aligned_alloc_shared<int>(128, Size, Q);
  int *ResA = aligned_alloc_shared<int>(128, Size, Q);
  int *ResB = aligned_alloc_shared<int>(128, Size, Q);
  std::fill(A, A + Size, 0);
  std::fill(B, B + Size, 0);
  std::fill(ResA, ResA + Size, 0);
  std::fill(ResB, ResB + Size, 0);

  try {
    Q.submit([&](handler &h) {
       range<1> GlobalRange(Size / SIMDSize);
       range<1> LocalRange(LocalSize);
       // There are many pairs of threads.
       // 0-th thread:
       //     A = 5;
       //     lsc_fence;
       //     B = 1;
       // 1-st thread:
       //     BB = B; // Don't use infinite loop 'while(B == 0) {}' as
       //             // it was in the original example at wiki-page.
       //             // That loop had incorrect assumption about compiler
       //             // would keep the load inside the loop.
       //     lsc_fence;
       //     AA = A; // If B is 1, then A must be equal to 5.
       h.parallel_for(nd_range<1>{GlobalRange, LocalRange},
                      [=](nd_item<1> NdId) SYCL_ESIMD_KERNEL {
                        auto GID = NdId.get_global_linear_id();
                        auto Offset = GID / 2 * SIMDSize;
                        auto ByteOffset = Offset * sizeof(int);

                        if (NdId.get_local_linear_id() % 2 == OddOrEven) {
                          // First thread: write data and condition
                          // and provoke gpu to reorder instructions
                          lsc_block_store<int, SIMDSize>(
                              A + Offset, simd<int, SIMDSize>{5});

                          // Protect from reordering the writes to A and B.
                          lsc_fence<lsc_memory_kind::untyped_global>();

                          lsc_block_store<int, SIMDSize>(
                              B + Offset, simd<int, SIMDSize>{1});
                        } else {
                          auto BVec = lsc_block_load<int, SIMDSize>(B + Offset);

                          // Protect from reordering the reads from B and A.
                          lsc_fence<lsc_memory_kind::untyped_global>();

                          auto AVec = lsc_block_load<int, SIMDSize>(A + Offset);

                          AVec.copy_to(ResA + Offset);
                          BVec.copy_to(ResB + Offset);
                        }
                      });
     }).wait();
  } catch (sycl::exception e) {
    std::cout << "\nSYCL exception caught: " << e.what();
    NumErrors = 1000;
  }

  for (int I = 0; I < Size / 2 && NumErrors < 10; I++) {
    if (ResB[I] != 0 && ResA[I] == 0) {
      NumErrors++;
      std::cout << I << ": Error - B was written before A: A = " << ResA[I]
                << ", B = " << ResB[I] << std::endl;
    }
  }

  free(A, Q);
  free(B, Q);
  free(ResA, Q);
  free(ResB, Q);
  std::cout << " Done" << (NumErrors ? " with ERRORS" : "") << std::endl;
  return NumErrors == 0;
}

bool testLocal(queue &Q) {
  constexpr size_t LocalSize = 16;
  constexpr unsigned int SIMDSize = 8;
  constexpr size_t Size = LocalSize * SIMDSize * 64;

  std::cout << "Running testLocal()...";

  int NumErrors = 0;
  int *ResA = aligned_alloc_shared<int>(128, Size, Q);
  int *ResB = aligned_alloc_shared<int>(128, Size, Q);
  std::fill(ResA, ResA + Size, 0);
  std::fill(ResB, ResB + Size, 0);

  try {
    Q.submit([&](handler &h) {
       range<2> GlobalRange(Size / SIMDSize, 2);
       range<2> LocalRange(LocalSize, 2);
       // There are many pairs of threads.
       // 0-th thread:
       //     A_SLM = 5;
       //     mem_fence
       //     B_SLM = 1;
       // 1-st thread:
       //     BB = B_SLM;
       //     mem_fence
       //     AA = A_SLM; // If BB is 1, then AA must be 5.
       h.parallel_for(
           nd_range<2>{GlobalRange, LocalRange},
           [=](nd_item<2> NdId) SYCL_ESIMD_KERNEL {
             constexpr int SLMSize = LocalSize * SIMDSize * 2 * sizeof(int);
             // Allocate SLM memory and initialize it with zeroes.
             slm_init<SLMSize>();
             if (NdId.get_local_linear_id() == 0) {
               for (int I = 0; I < SLMSize; I += SIMDSize * sizeof(int))
                 lsc_slm_block_store<int, SIMDSize>(I, simd<int, SIMDSize>(0));
             }
             barrier();

             auto Offset = NdId.get_local_id(0) * SIMDSize;
             auto ByteOffsetA = Offset * sizeof(int);
             auto ByteOffsetB = SLMSize / 2 + ByteOffsetA;
             if (NdId.get_local_id(1) == 0) {
               lsc_slm_block_store<int, SIMDSize>(ByteOffsetA,
                                                  simd<int, SIMDSize>(5));
               lsc_fence<lsc_memory_kind::shared_local>();
               lsc_slm_block_store<int, SIMDSize>(ByteOffsetB,
                                                  simd<int, SIMDSize>(1));
             } else {
               simd<int, SIMDSize> BVec =
                   lsc_slm_block_load<int, SIMDSize>(ByteOffsetB);
               lsc_fence<lsc_memory_kind::shared_local>();
               simd<int, SIMDSize> AVec =
                   lsc_slm_block_load<int, SIMDSize>(ByteOffsetA);

               AVec.copy_to(ResA + Offset);
               BVec.copy_to(ResB + Offset);
             }
           });
     }).wait();
  } catch (sycl::exception e) {
    std::cout << "\nSYCL exception caught: " << e.what();
    NumErrors = 1000;
  }

  for (int I = 0; I < Size && NumErrors < 10; I++) {
    if (ResB[I] != 0 && ResA[I] == 0) {
      NumErrors++;
      std::cout << I << ": Error - B was written before A: A = " << ResA[I]
                << ", B = " << ResB[I] << std::endl;
    }
  }

  std::cout << " Done" << (NumErrors ? " with ERRORS" : "") << std::endl;

  free(ResA, Q);
  free(ResB, Q);
  return NumErrors == 0;
}

int main() {
  queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  bool Passed = true;
  Passed &= testGlobal(Q, 0);
  Passed &= testGlobal(Q, 1);

  Passed &= testLocal(Q);

  std::cout << (Passed != 0 ? "PASSED" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
