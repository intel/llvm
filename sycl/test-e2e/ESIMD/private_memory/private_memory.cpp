//==---------------- private_memory.cpp  - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: arch-intel_gpu_pvc

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// The test verifies that basic ESIMD API works properly with
// private memory allocated on stack.

#include "../esimd_test_utils.hpp"

#include <sycl/specialization_id.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

constexpr specialization_id<uint32_t> PrivateArrayLenSC(1);

template <typename T, int PrivateArrayLenConst, int FirstN, bool UseConstExpr>
ESIMD_NOINLINE bool test(queue Q, int PrivateArrayLen) {
  std::cout << "Testing T=" << esimd_test::type_name<T>()
            << " ArrLen=" << PrivateArrayLen << ", FirstN=" << FirstN
            << ", UseConstExpr=" << UseConstExpr << "...\n";

  int GlobalRange = 2;
  int Size = GlobalRange * PrivateArrayLenConst;

  auto DstAsIsUPtr = esimd_test::usm_malloc_shared<T>(Q, Size);
  auto DstOddPlus1UPtr = esimd_test::usm_malloc_shared<T>(Q, Size);
  auto DstAfterBlockStoreUPtr = esimd_test::usm_malloc_shared<T>(Q, Size);

  T *DstAsIs = DstAsIsUPtr.get();
  T *DstOddPlus1 = DstOddPlus1UPtr.get();
  T *DstAfterBlockStore = DstAfterBlockStoreUPtr.get();
  for (int I = 0; I < Size; I++) {
    DstAsIs[I] = DstOddPlus1[I] = DstAfterBlockStore[I] = 0;
  }

  T TOne = static_cast<T>(1);
  T TTen = static_cast<T>(10);

  Q.submit([&](sycl::handler &CGH) {
     CGH.set_specialization_constant<PrivateArrayLenSC>(PrivateArrayLen);
     CGH.parallel_for(
         GlobalRange, [=](id<1> Id, sycl::kernel_handler KH) SYCL_ESIMD_KERNEL {
           uint32_t ArrayLen;
           if constexpr (UseConstExpr) {
             // This declaration masks the declaration of PrivateArrayLen on
             // HOST.
             ArrayLen = KH.get_specialization_constant<PrivateArrayLenSC>();
           } else {
             // Simply use PrivateArrayLen declared/initialized on HOST.
             ArrayLen = PrivateArrayLen;
           }
           T *PrivateArray = (T *)__builtin_alloca_with_align(
               sizeof(T) * ArrayLen, sizeof(T) * 8 * 16);

           // Initialize private memory
           for (int I = 0; I < ArrayLen; I++) {
             simd<int, 1> IV(static_cast<int>(Id) * PrivateArrayLen + I);
             simd<T, 1> TV = IV;
             TV.copy_to(PrivateArray + I);
           }

           simd<T, PrivateArrayLenConst> BigVec(PrivateArray);
           BigVec.copy_to(DstAsIs + ArrayLen * Id);

           // Check that scatter() works fine.
           auto FirstNOdd = BigVec.template select<FirstN, 2>(1).read();
           FirstNOdd = FirstNOdd + simd<T, FirstN>(TOne);
           simd<int, FirstN> FirstNOddByteOffsets(sizeof(T), 2 * sizeof(T));
           scatter(PrivateArray, FirstNOddByteOffsets, FirstNOdd);

           simd<T, PrivateArrayLenConst> BigVecOddPlus1(PrivateArray);
           BigVecOddPlus1.copy_to(DstOddPlus1 + ArrayLen * Id);

           if constexpr (PrivateArrayLenConst > FirstN &&
                         FirstN * sizeof(T) >= 16 &&
                         FirstN * sizeof(T) <= 8 * 16) {
             // Check that block_store() works fine.
             BigVec.copy_to(PrivateArray);
             simd<T, FirstN> BigVecFirstN = BigVec.template select<FirstN, 1>();
             BigVecFirstN = BigVecFirstN * simd<T, FirstN>(TTen);
             block_store(PrivateArray, BigVecFirstN);

             simd<T, PrivateArrayLenConst> BigVecAfterBlockStore(PrivateArray);
             BigVecAfterBlockStore.copy_to(DstAfterBlockStore + ArrayLen * Id);
           }
         });
   }).wait();

  for (int I = 0; I < Size; I++) {
    T Expected = I;
    if (DstAsIs[I] != Expected) {
      std::cout << "Error/DstAsIs[" << I << "]: " << DstAsIs[I]
                << " != " << Expected << "(Expected)" << std::endl;
      return false;
    }

    int CurrentWI = I / PrivateArrayLenConst;
    int IndexInWI = I - CurrentWI * PrivateArrayLenConst;

    Expected = I;
    if ((IndexInWI & 1) && IndexInWI < FirstN * 2)
      Expected = I + 1;
    if (DstOddPlus1[I] != Expected) {
      std::cout << "Error/DstOddPlus1[" << I << "]: " << DstOddPlus1[I]
                << " != " << Expected << "(Expected)" << std::endl;
      return false;
    }

    if constexpr (PrivateArrayLenConst > FirstN && FirstN * sizeof(T) >= 16 &&
                  FirstN * sizeof(T) <= 8 * 16) {
      Expected = I;
      if (IndexInWI < FirstN)
        // Expected = I * I;
        Expected = (T)I * TTen;
      if (DstAfterBlockStore[I] != Expected) {
        std::cout << "Error/DstAfterBlockStore[" << I
                  << "]: " << DstAfterBlockStore[I] << " != " << Expected
                  << "(Expected)" << std::endl;
        return false;
      }
    }
  }

  return true;
}

template <typename T> bool tests(queue Q) {
  constexpr bool UseSpecConst = true;

  bool Passed = true;
  Passed &= test<T, 32, 16, UseSpecConst>(Q, 32);
  Passed &= test<T, 32, 8, !UseSpecConst>(Q, 32);

  Passed &= test<T, 256, 32, UseSpecConst>(Q, 256);
  Passed &= test<T, 256, 16, !UseSpecConst>(Q, 256);

  return Passed;
}

int main() {
  queue Q;
  std::cout << "Running on " << Q.get_device().get_info<info::device::name>()
            << "\n";

  bool Passed = true;
  Passed &= tests<int8_t>(Q);
  Passed &= tests<short>(Q);
  Passed &= tests<int>(Q);
  Passed &= tests<int64_t>(Q);

  Passed &= tests<float>(Q);
  if (Q.get_device().has(sycl::aspect::fp16))
    Passed &= tests<sycl::half>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Passed &= tests<double>(Q);

    // TODO: GPU driver reports an error during JIT compilation.
    //       Report and enable this case when driver is fixed.
    // Passed &= tests<sycl::ext::oneapi::bfloat16>(Q);

#ifdef TEST_TFLOAT32
  Passed &= tests<sycl::ext::intel::experimental::esimd::tfloat32>(Q);
#endif // TEST_TFLOAT32

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
