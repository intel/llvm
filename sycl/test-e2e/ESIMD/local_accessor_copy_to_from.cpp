//==-- local_accessor_copy_to_from.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES-INTEL-DRIVER: lin: 27202, win: 101.4677
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The test checks functionality of the gather/scatter local
// accessor-based ESIMD intrinsics.

#include "esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

constexpr uint32_t LocalRange = 16;
constexpr uint32_t GlobalRange = LocalRange * 2; // 2 groups.

template <typename T, unsigned VL> bool test(queue q) {
  constexpr size_t Size = VL * LocalRange;
  std::cout << "Running case: T=" << esimd_test::type_name<T>() << " VL=" << VL
            << std::endl;

  // The test is going to use (LocalRange * VL) elements of T type.
  auto Dev = q.get_device();
  auto DeviceSLMSize = Dev.get_info<sycl::info::device::local_mem_size>();
  if (DeviceSLMSize < Size * sizeof(T)) {
    // Report an error - the test needs a fix.
    std::cerr << "Error: Test needs more SLM memory than device has!"
              << std::endl;
    return false;
  }

  T *A = new T[GlobalRange * VL];

  for (unsigned i = 0; i < GlobalRange * VL; ++i) {
    A[i] = static_cast<T>(0);
  }

  try {
    buffer<T, 1> buf(A, range<1>(GlobalRange * VL));
    nd_range<1> NDRange{range<1>{GlobalRange}, range<1>{LocalRange}};
    q.submit([&](handler &CGH) {
       auto LocalAcc = local_accessor<T, 1>(Size, CGH);
       auto Acc = buf.template get_access<access::mode::read_write>(CGH);
       CGH.parallel_for(NDRange, [=](nd_item<1> Item) SYCL_ESIMD_KERNEL {
         uint32_t GID = Item.get_global_id(0);
         uint32_t LID = Item.get_local_id(0);

         simd<T, VL> ValuesToSLM(GID * 100, 1);
         ValuesToSLM.copy_to(LocalAcc, LID * VL * sizeof(T));

         Item.barrier();

         if (LID == 0) {
           for (int LID = 0; LID < LocalRange; LID++) {
             simd<T, VL> ValuesFromSLM;
             ValuesFromSLM.copy_from(LocalAcc, LID * VL * sizeof(T));
             ValuesFromSLM.copy_to(Acc, (GID + LID) * VL * sizeof(T));
           } // end for (int LID = 0; LID < LocalRange; LID++)
         }   // end if (LID == 0)
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    return false;
  }

  bool Pass = true;
  for (int I = 0; I < GlobalRange * VL; I++) {
    int GID = I / VL;
    int LID = GID % LocalRange;
    int VecElementIndex = I % VL;

    T Expected = GID * 100 + VecElementIndex;
    T Computed = A[I];
    if (Computed != Expected) {
      std::cout << "Error: Out[" << I << "]:" << Computed << " != " << Expected
                << ":[expected]" << std::endl;
      Pass = false;
    }
  }

  delete[] A;

  return Pass;
}

int main() {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<char, 1>(q);
  passed &= test<char, 2>(q);
  passed &= test<char, 4>(q);
  passed &= test<char, 8>(q);
  passed &= test<char, 16>(q);
  passed &= test<char, 32>(q);
  passed &= test<char, 64>(q);
  passed &= test<char, 128>(q);
  passed &= test<short, 1>(q);
  passed &= test<short, 2>(q);
  passed &= test<short, 4>(q);
  passed &= test<short, 8>(q);
  passed &= test<short, 16>(q);
  passed &= test<short, 32>(q);
  passed &= test<short, 64>(q);
  passed &= test<short, 128>(q);
  passed &= test<int, 1>(q);
  passed &= test<int, 2>(q);
  passed &= test<int, 4>(q);
  passed &= test<int, 8>(q);
  passed &= test<int, 16>(q);
  passed &= test<int, 32>(q);
  passed &= test<int, 64>(q);
  passed &= test<int, 128>(q);
  passed &= test<float, 1>(q);
  passed &= test<float, 2>(q);
  passed &= test<float, 4>(q);
  passed &= test<float, 8>(q);
  passed &= test<float, 16>(q);
  passed &= test<float, 32>(q);
  passed &= test<float, 64>(q);
  passed &= test<float, 128>(q);
  return passed ? 0 : 1;
}
