//==------------ fence.cpp - DPC++ ESIMD on-device test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 || arch-intel_gpu_pvc
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the intrinsic fence.
// It is based on https://en.wikipedia.org/wiki/Memory_barrier#Example

#include <cmath>
#include <numeric>

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

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
       //     fence;
       //     B = 1;
       // 1-st thread:
       //     BB = B; // Don't use infinite loop 'while(B == 0) {}' as
       //             // it was in the original example at wiki-page.
       //             // That loop had incorrect assumption about compiler
       //             // would keep the load inside the loop.
       //     fence;
       //     AA = A; // If B is 1, then A must be equal to 5.
       h.parallel_for(nd_range<1>{GlobalRange, LocalRange},
                      [=](nd_item<1> NdId) SYCL_ESIMD_KERNEL {
                        auto GID = NdId.get_global_linear_id();
                        auto Offset = GID / 2 * SIMDSize;
                        auto ByteOffset = Offset * sizeof(int);

                        if (NdId.get_local_linear_id() % 2 == OddOrEven) {
                          // First thread: write data and condition
                          // and provoke gpu to reorder instructions
                          block_store(A + Offset, simd<int, SIMDSize>{5});

                          // Protect from reordering the writes to A and B.
                          fence<memory_kind::global>();

                          block_store(B + Offset, simd<int, SIMDSize>{1});
                        } else {
                          auto BVec = block_load<int, SIMDSize>(B + Offset);

                          // Protect from reordering the reads from B and A.
                          fence<memory_kind::global>();

                          auto AVec = block_load<int, SIMDSize>(A + Offset);

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
                 slm_block_store(I, simd<int, SIMDSize>(0));
             }
             barrier();

             auto Offset = NdId.get_local_id(0) * SIMDSize;
             auto ByteOffsetA = Offset * sizeof(int);
             auto ByteOffsetB = SLMSize / 2 + ByteOffsetA;
             if (NdId.get_local_id(1) == 0) {
               slm_block_store(ByteOffsetA, simd<int, SIMDSize>(5));
               fence<memory_kind::local>();
               slm_block_store(ByteOffsetB, simd<int, SIMDSize>(1));
             } else {
               auto BVec = slm_block_load<int, SIMDSize>(ByteOffsetB);
               fence<memory_kind::local>();
               auto AVec = slm_block_load<int, SIMDSize>(ByteOffsetA);

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
  auto Q = queue{gpu_selector_v};
  esimd_test::printTestLabel(Q);

  bool Passed = true;
  Passed &= testGlobal(Q, 0);
  Passed &= testGlobal(Q, 1);

  Passed &= testLocal(Q);

  std::cout << (Passed != 0 ? "PASSED" : "FAILED") << std::endl;
  return Passed ? 0 : 1;
}
