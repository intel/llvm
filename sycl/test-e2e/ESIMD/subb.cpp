//==---------------- subb.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies ESIMD API that substracts 2 32-bit integer scalars/vectors
// with borrow returning the result as 2 parts: borrow flag the input modified
// operand and substraction result as return from function.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace iesimd = sycl::ext::intel::experimental::esimd;

template <int N, bool AIsVector, bool BIsVector> bool test(sycl::queue Q) {
  static_assert(AIsVector || BIsVector || N == 1,
                "(Scalar - Scalar) case must have N==1");

  uint32_t ValuesToTryHost[] = {0,
                                1,
                                static_cast<uint32_t>(-1),
                                0x7f,
                                static_cast<uint32_t>(-0x7f),
                                0x7fff,
                                static_cast<uint32_t>(-0x7fff),
                                0x7ffff,
                                static_cast<uint32_t>(-0x7ffff),
                                0x7ffffff,
                                0x80,
                                static_cast<uint32_t>(-0x80),
                                0x8000,
                                static_cast<uint32_t>(-0x8000),
                                0x800000,
                                static_cast<uint32_t>(-0x800000),
                                0x80000000};
  ;
  uint32_t ValuesToTrySize = sizeof(ValuesToTryHost) / sizeof(uint32_t);

  std::cout << "Running case:  N=" << N << ", AIsVector=" << AIsVector
            << ", BIsVector=" << BIsVector << std::endl;

  auto ValuesToTryUPtr =
      esimd_test::usm_malloc_shared<uint32_t>(Q, ValuesToTrySize);
  uint32_t *ValuesToTryPtr = ValuesToTryUPtr.get();
  memcpy(ValuesToTryPtr, ValuesToTryHost, ValuesToTrySize * sizeof(uint32_t));

  auto ResultsMatrixUPtr = esimd_test::usm_malloc_shared<uint32_t>(
      Q, ValuesToTrySize * ValuesToTrySize * N);
  auto BorrowMatrixUPtr = esimd_test::usm_malloc_shared<uint32_t>(
      Q, ValuesToTrySize * ValuesToTrySize * N);
  uint32_t *ResultsMatrixPtr = ResultsMatrixUPtr.get();
  uint32_t *BorrowMatrixPtr = BorrowMatrixUPtr.get();

  try {
    Q.single_task([=]() SYCL_ESIMD_KERNEL {
       simd<uint32_t, N> VecInc(0, 1);
       for (int AI = 0; AI < ValuesToTrySize; AI++) {
         using AType =
             std::conditional_t<AIsVector, simd<uint32_t, N>, uint32_t>;
         uint32_t AScalar = simd<uint32_t, 1>(
             reinterpret_cast<uint32_t *>(ValuesToTryPtr) + AI)[0];
         AType A = AScalar;
         if constexpr (AIsVector)
           A += VecInc;

         for (int BI = 0; BI < ValuesToTrySize; BI++) {
           using BType =
               std::conditional_t<BIsVector, simd<uint32_t, N>, uint32_t>;
           uint32_t BScalar = simd<uint32_t, 1>(
               reinterpret_cast<uint32_t *>(ValuesToTryPtr) + BI)[0];
           BType B = BScalar;
           if constexpr (BIsVector)
             B += VecInc;

           using ResType = std::conditional_t<AIsVector || BIsVector,
                                              simd<uint32_t, N>, uint32_t>;
           ResType Borrow = 0;
           ResType Res = iesimd::subb(Borrow, A, B);

           if constexpr (AIsVector || BIsVector) {
             Borrow.copy_to(BorrowMatrixPtr + (ValuesToTrySize * AI + BI) * N);
             Res.copy_to(ResultsMatrixPtr + (ValuesToTrySize * AI + BI) * N);
           } else {
             simd<uint32_t, 1> Borrow1 = Borrow;
             simd<uint32_t, 1> Res1 = Res;
             Borrow1.copy_to(BorrowMatrixPtr + (ValuesToTrySize * AI + BI) * N);
             Res1.copy_to(ResultsMatrixPtr + (ValuesToTrySize * AI + BI) * N);
           }

         } // end for BI
       }   // end for AI
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  using Result64T = uint64_t;
  int NumErrors = 0;
  for (int AI = 0; AI < ValuesToTrySize; AI++) {
    for (int BI = 0; BI < ValuesToTrySize; BI++) {
      for (int I = 0; I < N; I++) {
        uint32_t A = ValuesToTryHost[AI];
        if constexpr (AIsVector)
          A += I;
        uint32_t B = ValuesToTryHost[BI];
        if constexpr (BIsVector)
          B += I;
        Result64T R = static_cast<uint32_t>(A);
        R -= static_cast<uint32_t>(B);

        uint32_t ExpectedRes = R & 0xffffffff;
        uint32_t ExpectedBorrow = A < B;
        uint32_t ComputedRes =
            ResultsMatrixPtr[(AI * ValuesToTrySize + BI) * N + I];
        uint32_t ComputedBorrow =
            BorrowMatrixPtr[(AI * ValuesToTrySize + BI) * N + I];

        if (ComputedRes != ExpectedRes || ComputedBorrow != ExpectedBorrow) {
          std::cout << "Error for (" << AI << "," << BI << "): " << std::hex
                    << A << " - " << B << " is Computed(" << ComputedBorrow
                    << "," << ComputedRes << ") != Expected (" << ExpectedBorrow
                    << "," << ExpectedRes << "), R = " << R << std::dec << "\n";
          NumErrors++;
        }
      }
    }
  }

  return NumErrors == 0;
}

int main() {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto D = Q.get_device();
  std::cout << "Running on " << D.get_info<info::device::name>() << "\n";

  constexpr bool AIsVector = true;
  constexpr bool BIsVector = true;
  bool Pass = true;
  Pass &= test<16, AIsVector, BIsVector>(Q);
  Pass &= test<8, AIsVector, !BIsVector>(Q);
  Pass &= test<4, !AIsVector, BIsVector>(Q);

  Pass &= test<1, AIsVector, BIsVector>(Q);
  Pass &= test<1, AIsVector, !BIsVector>(Q);
  Pass &= test<1, !AIsVector, BIsVector>(Q);

  Pass &= test<1, !AIsVector, !BIsVector>(Q);

  std::cout << (Pass > 0 ? "Passed\n" : "FAILED\n");
  return Pass ? 0 : 1;
}
