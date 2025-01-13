//==---------------- addc.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/14868

// The test verifies ESIMD API that adds 2 32-bit integer scalars/vectors with
// carry returning the result as 2 parts: carry flag the input modified operand
// and addition result as return from function.

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename T, int N, bool AIsVector, bool BIsVector>
bool test(sycl::queue Q) {
  static_assert(AIsVector || BIsVector || N == 1,
                "(Scalar + Scalar) case must have N==1");
  uint32_t ValuesToTryHost32[] = {0,
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

  uint64_t ValuesToTryHost64[] = {0,
                                  1,
                                  static_cast<uint64_t>(-1),
                                  0x7f,
                                  static_cast<uint64_t>(-0x7f),
                                  0x7fff,
                                  static_cast<uint64_t>(-0x7fff),
                                  0x7ffff,
                                  static_cast<uint64_t>(-0x7ffff),
                                  0x7ffffff,
                                  static_cast<uint64_t>(-0x7ffffff),
                                  0x7ffffffff,
                                  static_cast<uint64_t>(-0x7ffffffff),
                                  0x80,
                                  static_cast<uint64_t>(-0x80),
                                  0x8000,
                                  static_cast<uint64_t>(-0x8000),
                                  0x800000,
                                  static_cast<uint64_t>(-0x800000),
                                  0x80000000,
                                  static_cast<uint64_t>(-0x80000000),
                                  0x8000000000,
                                  static_cast<uint64_t>(-0x8000000000)};

  uint32_t ValuesToTrySize = 0;
  if constexpr (sizeof(T) == 4) {
    ValuesToTrySize = sizeof(ValuesToTryHost32) / sizeof(T);
  } else if constexpr (sizeof(T) == 8) {
    ValuesToTrySize = sizeof(ValuesToTryHost64) / sizeof(T);
  }

  std::cout << "Running case: T=" << esimd_test::type_name<T>() << " N = " << N
            << ", AIsVector = " << AIsVector << ", BIsVector=" << BIsVector
            << std::endl;

  auto ValuesToTryUPtr = esimd_test::usm_malloc_shared<T>(Q, ValuesToTrySize);
  T *ValuesToTryPtr = ValuesToTryUPtr.get();
  if constexpr (sizeof(T) == 4) {
    memcpy(ValuesToTryPtr, ValuesToTryHost32, ValuesToTrySize * sizeof(T));
  } else if constexpr (sizeof(T) == 8) {
    memcpy(ValuesToTryPtr, ValuesToTryHost64, ValuesToTrySize * sizeof(T));
  }

  auto ResultsMatrixUPtr = esimd_test::usm_malloc_shared<T>(
      Q, ValuesToTrySize * ValuesToTrySize * N);
  auto CarryMatrixUPtr = esimd_test::usm_malloc_shared<T>(
      Q, ValuesToTrySize * ValuesToTrySize * N);
  T *ResultsMatrixPtr = ResultsMatrixUPtr.get();
  T *CarryMatrixPtr = CarryMatrixUPtr.get();

  try {
    Q.single_task([=]() SYCL_ESIMD_KERNEL {
       simd<T, N> VecInc(0, 1);
       for (int AI = 0; AI < ValuesToTrySize; AI++) {
         using AType = std::conditional_t<AIsVector, simd<T, N>, T>;
         T AScalar = simd<T, 1>(reinterpret_cast<T *>(ValuesToTryPtr) + AI)[0];
         AType A = AScalar;
         if constexpr (AIsVector)
           A += VecInc;

         for (int BI = 0; BI < ValuesToTrySize; BI++) {
           using BType = std::conditional_t<BIsVector, simd<T, N>, T>;
           T BScalar =
               simd<T, 1>(reinterpret_cast<T *>(ValuesToTryPtr) + BI)[0];
           BType B = BScalar;
           if constexpr (BIsVector)
             B += VecInc;

           using ResType =
               std::conditional_t<AIsVector || BIsVector, simd<T, N>, T>;
           ResType Carry = 0;
           ResType Res = addc(Carry, A, B);

           if constexpr (AIsVector || BIsVector) {
             Carry.copy_to(CarryMatrixPtr + (ValuesToTrySize * AI + BI) * N);
             Res.copy_to(ResultsMatrixPtr + (ValuesToTrySize * AI + BI) * N);
           } else {
             simd<T, 1> Carry1 = Carry;
             simd<T, 1> Res1 = Res;
             Carry1.copy_to(CarryMatrixPtr + (ValuesToTrySize * AI + BI) * N);
             Res1.copy_to(ResultsMatrixPtr + (ValuesToTrySize * AI + BI) * N);
           }

         } // end for BI
       } // end for AI
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 0;
  }

  using ResultT = std::conditional_t<
      2 * sizeof(T) == 8, uint64_t,
      std::conditional_t<2 * sizeof(T) == 16, __uint128_t, T>>;

  int NumErrors = 0;
  for (int AI = 0; AI < ValuesToTrySize; AI++) {
    for (int BI = 0; BI < ValuesToTrySize; BI++) {
      for (int I = 0; I < N; I++) {
        T A = ValuesToTryPtr[AI];
        if constexpr (AIsVector)
          A += I;
        T B = ValuesToTryPtr[BI];
        if constexpr (BIsVector)
          B += I;
        ResultT R = static_cast<T>(A);
        R += static_cast<T>(B);

        T ExpectedRes = R & ~(T)(0);
        T ExpectedCarry = (R >> (8 * sizeof(T))) & ~(T)(0);
        T ComputedRes = ResultsMatrixPtr[(AI * ValuesToTrySize + BI) * N + I];
        T ComputedCarry = CarryMatrixPtr[(AI * ValuesToTrySize + BI) * N + I];
        if (ComputedRes != ExpectedRes || ComputedCarry != ExpectedCarry) {
          std::cout << "Error for (" << AI << "," << BI << "): " << A << " + "
                    << B << " is Computed(" << ComputedCarry << ","
                    << ComputedRes << ") != Expected (" << ExpectedCarry << ","
                    << ExpectedRes << ")"
                    << "\n";
          NumErrors++;
        }
      }
    }
  }

  return NumErrors == 0;
}

template <typename T> bool test(sycl::queue Q) {
  constexpr bool AIsVector = true;
  constexpr bool BIsVector = true;
  bool Pass = true;
  Pass &= test<T, 16, AIsVector, BIsVector>(Q);
  Pass &= test<T, 8, AIsVector, !BIsVector>(Q);
  Pass &= test<T, 4, !AIsVector, BIsVector>(Q);

  Pass &= test<T, 1, AIsVector, BIsVector>(Q);
  Pass &= test<T, 1, AIsVector, !BIsVector>(Q);
  Pass &= test<T, 1, !AIsVector, BIsVector>(Q);

  Pass &= test<T, 1, !AIsVector, !BIsVector>(Q);
  return Pass;
}

int main() {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(Q);
  bool Pass = true;

  Pass &= test<uint32_t>(Q);
  Pass &= test<uint64_t>(Q);

  std::cout << (Pass > 0 ? "Passed\n" : "FAILED\n");
  return Pass ? 0 : 1;
}
