//==---------------- imulh_umulh.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The test verifies ESIMD API that multiplies 2 32-bit integer scalars/vectors
// resulting into 64-bit result and returning the result as 2 parts:
// lower 32-bits in the input modified operand and upper 32-bits as return
// from function.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace iesimd = sycl::ext::intel::experimental::esimd;

template <typename RT, typename T0, typename T1, int N, bool AIsVector,
          bool BIsVector>
bool test(sycl::queue Q) {
  static_assert(AIsVector || BIsVector || N == 1,
                "(Scalar * Scalar) case must have N==1");

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
  uint32_t ValuesToTrySize = sizeof(ValuesToTryHost) / sizeof(uint32_t);

  std::cout << "Running case: RT=" << esimd_test::type_name<RT>()
            << ", T0=" << esimd_test::type_name<T0>()
            << ", T1=" << esimd_test::type_name<T1>() << ", N=" << N
            << ", AIsVector=" << AIsVector << ", BIsVector=" << BIsVector
            << std::endl;

  auto ValuesToTryUPtr =
      esimd_test::usm_malloc_shared<uint32_t>(Q, ValuesToTrySize);
  uint32_t *ValuesToTryPtr = ValuesToTryUPtr.get();
  memcpy(ValuesToTryPtr, ValuesToTryHost, ValuesToTrySize * sizeof(uint32_t));

  auto ResultsMatrixLoUPtr = esimd_test::usm_malloc_shared<RT>(
      Q, ValuesToTrySize * ValuesToTrySize * N);
  auto ResultsMatrixHiUPtr = esimd_test::usm_malloc_shared<RT>(
      Q, ValuesToTrySize * ValuesToTrySize * N);
  RT *ResultsMatrixLoPtr = ResultsMatrixLoUPtr.get();
  RT *ResultsMatrixHiPtr = ResultsMatrixHiUPtr.get();

  try {
    Q.single_task([=]() SYCL_ESIMD_KERNEL {
       simd<T0, N> VecInc(0, 1);
       for (int AI = 0; AI < ValuesToTrySize; AI++) {
         using AType = std::conditional_t<AIsVector, simd<T0, N>, T0>;
         T0 AScalar =
             simd<T0, 1>(reinterpret_cast<T0 *>(ValuesToTryPtr) + AI)[0];
         AType A = AScalar;
         if constexpr (AIsVector)
           A += VecInc;

         for (int BI = 0; BI < ValuesToTrySize; BI++) {
           using BType = std::conditional_t<BIsVector, simd<T1, N>, T1>;
           T1 BScalar =
               simd<T1, 1>(reinterpret_cast<T1 *>(ValuesToTryPtr) + BI)[0];
           BType B = BScalar;
           if constexpr (BIsVector)
             B += VecInc;

           using ResType =
               std::conditional_t<AIsVector || BIsVector, simd<RT, N>, RT>;
           ResType ResLo;
           ResType ResHi = iesimd::imul(ResLo, A, B);

           if constexpr (AIsVector || BIsVector) {
             ResLo.copy_to(ResultsMatrixLoPtr +
                           (ValuesToTrySize * AI + BI) * N);
             ResHi.copy_to(ResultsMatrixHiPtr +
                           (ValuesToTrySize * AI + BI) * N);
           } else {
             simd<RT, 1> ResLo1 = ResLo;
             simd<RT, 1> ResHi1 = ResHi;
             ResLo1.copy_to(ResultsMatrixLoPtr +
                            (ValuesToTrySize * AI + BI) * N);
             ResHi1.copy_to(ResultsMatrixHiPtr +
                            (ValuesToTrySize * AI + BI) * N);
           }

         } // end for BI
       }   // end for AI
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  using Common32T = decltype(std::declval<T0>() * std::declval<T1>());
  using Result64T = std::conditional_t<std::is_signed_v<RT>, int64_t, uint64_t>;
  int NumErrors = 0;
  for (int AI = 0; AI < ValuesToTrySize; AI++) {
    for (int BI = 0; BI < ValuesToTrySize; BI++) {
      for (int I = 0; I < N; I++) {
        T0 A = ValuesToTryHost[AI];
        if constexpr (AIsVector)
          A += I;
        T1 B = ValuesToTryHost[BI];
        if constexpr (BIsVector)
          B += I;
        Result64T R = static_cast<Common32T>(A);
        R *= static_cast<Common32T>(B);

        RT ExpectedResLo = R & 0xffffffff;
        RT ExpectedResHi = (R >> 32) & 0xffffffff;
        RT ComputedResLo =
            ResultsMatrixLoPtr[(AI * ValuesToTrySize + BI) * N + I];
        RT ComputedResHi =
            ResultsMatrixHiPtr[(AI * ValuesToTrySize + BI) * N + I];
        if (ComputedResLo != ExpectedResLo || ComputedResHi != ExpectedResHi) {
          std::cout << "Error for (" << AI << "," << BI << "): " << A << "x"
                    << B << " is Computed(" << ComputedResHi << ","
                    << ComputedResLo << ") != Expected (" << ExpectedResHi
                    << "," << ExpectedResLo << "), R = " << R << "\n";
          NumErrors++;
        }
      }
    }
  }

  return NumErrors == 0;
}

template <int N, bool AIsVector, bool BIsVector> bool tests(sycl::queue Q) {
  bool Pass = true;
  Pass &= test<uint32_t, uint32_t, uint32_t, N, AIsVector, BIsVector>(Q);
  Pass &= test<uint32_t, uint32_t, int32_t, N, AIsVector, BIsVector>(Q);
  Pass &= test<uint32_t, int32_t, uint32_t, N, AIsVector, BIsVector>(Q);
  Pass &= test<uint32_t, int32_t, int32_t, N, AIsVector, BIsVector>(Q);
  return Pass;
}

int main() {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto D = Q.get_device();
  std::cout << "Running on " << D.get_info<info::device::name>() << "\n";

  constexpr bool AIsVector = true;
  constexpr bool BIsVector = true;
  bool Pass = true;
  Pass &= tests<16, AIsVector, BIsVector>(Q);
  Pass &= tests<8, AIsVector, !BIsVector>(Q);
  Pass &= tests<4, !AIsVector, BIsVector>(Q);

  Pass &= tests<1, AIsVector, BIsVector>(Q);
  Pass &= tests<1, AIsVector, !BIsVector>(Q);
  Pass &= tests<1, !AIsVector, BIsVector>(Q);

  Pass &= tests<1, !AIsVector, !BIsVector>(Q);

  std::cout << (Pass > 0 ? "Passed\n" : "FAILED\n");
  return Pass ? 0 : 1;
}
