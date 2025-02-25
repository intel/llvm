//==-- local_accessor_block_load_store.cpp  - DPC++ ESIMD on-device test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 27202, win: 101.4677
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// This test verifies usage of block_load/block_store for local_accessor.

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename T, int VL, int Align = 16> bool test(queue Q) {
  std::cout << "Running case: T=" << esimd_test::type_name<T>() << ", VL=" << VL
            << ", Align=" << Align << std::endl;

  constexpr uint32_t LocalRange = 16;
  constexpr uint32_t GlobalRange = LocalRange * 2; // 2 groups.

  // The test is going to use (LocalRange * VL) elements of T type.
  auto Dev = Q.get_device();
  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  constexpr uint32_t UsedSLMSize = LocalRange * VL * sizeof(T) + Align;
  if (DeviceSLMSize < UsedSLMSize) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has!"
              << std::endl;
    return false;
  }

  T *Out = malloc_shared<T>(GlobalRange * VL, Q);
  for (int I = 0; I < GlobalRange * VL; I++)
    Out[I] = -1;

  try {
    nd_range<1> NDRange{range<1>{GlobalRange}, range<1>{LocalRange}};
    Q.submit([&](handler &CGH) {
       auto LocalAcc = local_accessor<T, 1>(UsedSLMSize / sizeof(T), CGH);

       CGH.parallel_for(NDRange, [=](nd_item<1> Item) SYCL_ESIMD_KERNEL {
         uint32_t GID = Item.get_global_id(0);
         uint32_t LID = Item.get_local_id(0);
         overaligned_tag<Align> AlignTag;

         simd<int, VL> IntValues(GID * 100, 1);
         simd<T, VL> ValuesToSLM = IntValues;
         block_store(LocalAcc, Align + LID * VL * sizeof(T), ValuesToSLM,
                     AlignTag);

         Item.barrier();

         if (LID == 0) {
           for (int LID = 0; LID < LocalRange; LID++) {
             simd<T, VL> ValuesFromSLM = block_load<T, VL>(
                 LocalAcc, Align + LID * VL * sizeof(T), AlignTag);
             ValuesFromSLM.copy_to(Out + (GID + LID) * VL);
           } // end for (int LID = 0; LID < LocalRange; LID++)
         }   // end if (LID == 0)
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(Out, Q);
    return false;
  }

  bool Pass = true;
  for (int I = 0; I < GlobalRange * VL; I++) {
    int GID = I / VL;
    int LID = GID % LocalRange;
    int VecElementIndex = I % VL;

    T Expected = GID * 100 + VecElementIndex;
    T Computed = Out[I];
    if (Computed != Expected) {
      std::cout << "Error: Out[" << I << "]:" << Computed << " != " << Expected
                << ":[expected]" << std::endl;
      Pass = false;
    }
  }

  free(Out, Q);
  return Pass;
}

int main() {
  auto Q = queue{gpu_selector_v};
  auto Dev = Q.get_device();
  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  esimd_test::printTestLabel(Q, "Local memory size available", DeviceSLMSize);

  constexpr size_t Align4 = 4;
  constexpr size_t Align8 = 8;
  constexpr size_t Align16 = 16;

  bool Pass = true;
  Pass &= test<int, 16, Align16>(Q);
  Pass &= test<float, 16, Align16>(Q);

  if (Dev.has(aspect::fp16))
    Pass &= test<sycl::half, 16, Align16>(Q);

  // Check SLM load/store with vector size that is not power of 2
  // and/or is too big for 1 flat-load/store.
  Pass &= test<int, 16, Align4>(Q);
  Pass &= test<float, 16, Align8>(Q);

  std::cout << "Test result: " << (Pass ? "Pass" : "Fail") << std::endl;
  return Pass ? 0 : 1;
}
