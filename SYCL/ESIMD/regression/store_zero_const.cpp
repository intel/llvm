//==---------- store_zero_const.cpp - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This is a regression for vector BE bug:
// storing a constant zero through an USM pointer causes BE crash with
// "error: unsupported type for load/store" message.
// Use -fsycl-device-code-split=per_kernel so that each kernel's compilation
// does not affect others.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cstdint>
#include <iostream>

using namespace cl::sycl;

template <typename T> bool test(queue Q) {
  std::cout << "  Testing " << typeid(T).name() << "...\n";

  constexpr unsigned Size = 1024;
  constexpr unsigned VL = 32;
  constexpr unsigned GroupSize = 8;

  T *A = malloc_shared<T>(Size * sizeof(T), Q);

  range<1> GlobalRange{Size / VL};
  range<1> LocalRange{GroupSize};
  nd_range<1> Range(GlobalRange, LocalRange);

#define GOLD_VAL 0

  try {
    auto E = Q.submit([&](handler &CGH) {
      CGH.parallel_for<T>(Range, [=](nd_item<1> Ndi) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        int I = Ndi.get_global_id(0);
        for (auto Ind = 0; Ind < VL; ++Ind) {
          A[I * VL + Ind] = static_cast<T>(GOLD_VAL);
        }
      });
    });
    E.wait();
  } catch (sycl::exception const &E) {
    std::cout << "SYCL exception caught: " << E.what() << '\n';
    free(A, Q);
    return false; // not success
  }
  int ErrCnt = 0;

  for (unsigned I = 0; I < Size; ++I) {
    if (A[I] != GOLD_VAL) {
      if (++ErrCnt < 10) {
        std::cout << "failed at index " << I << ": " << A[I] << "\n";
      }
    }
  }
  if (ErrCnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - ErrCnt) / (float)Size) * 100.0f << "% ("
              << (Size - ErrCnt) << "/" << Size << ")\n";
  }
  std::cout << (ErrCnt > 0 ? "FAILED\n" : "Passed\n");
  sycl::free(A, Q);

  return ErrCnt == 0;
}

int main(void) {
  queue Q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << "\n";

  bool Passed = 1;
  Passed &= test<unsigned char>(Q);
  Passed &= test<char>(Q);
  Passed &= test<unsigned short>(Q);
  Passed &= test<short>(Q);
  Passed &= test<unsigned int>(Q);
  Passed &= test<uint64_t>(Q);
  Passed &= test<int64_t>(Q);
  Passed &= test<float>(Q);
  Passed &= test<double>(Q);

  std::cout << (Passed ? "Passed\n" : "FAILED\n");
  return Passed ? 0 : 1;
}
