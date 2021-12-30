//==---- svm_gather_scatter.cpp  - DPC++ ESIMD svm gather/scatter test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Regression test for SVM gather/scatter API.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/builtins_esimd.hpp>
#include <algorithm>
#include <array>
#include <iostream>

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T, int N> bool test(queue &Q) {
  std::cout << "  Running " << typeid(T).name() << " test, N=" << N << "...\n";

  struct Deleter {
    queue Q;
    void operator()(T *Ptr) {
      if (Ptr) {
        sycl::free(Ptr, Q);
      }
    }
  };

  std::unique_ptr<T, Deleter> Ptr1(sycl::malloc_shared<T>(N, Q), Deleter{Q});
  std::unique_ptr<T, Deleter> Ptr2(sycl::malloc_shared<T>(N, Q), Deleter{Q});

  T *Src = Ptr1.get();
  T *Dst = Ptr2.get();

  for (int I = 0; I < N; ++I) {
    Src[I] = I + 1;
    Dst[I] = 0;
  }

  try {
    Q.submit([&](handler &CGH) {
       CGH.parallel_for(sycl::range<1>{1}, [=](id<1>) SYCL_ESIMD_KERNEL {
         simd<uint32_t, N> Offsets(0u, sizeof(T));
         scatter<T, N>(Dst, Offsets, gather<T, N>(Src, Offsets));
       });
     }).wait();
  } catch (cl::sycl::exception const &E) {
    std::cout << "ERROR. SYCL exception caught: " << E.what() << std::endl;
    return false;
  }

  unsigned NumErrs = 0;
  for (int I = 0; I < N; ++I)
    if (Dst[I] != Src[I])
      if (++NumErrs <= 10)
        std::cout << "failed at " << I << ": " << Dst[I]
                  << " (Dst) != " << Src[I] << " (Src)\n";

  std::cout << (NumErrs == 0 ? "    Passed\n" : "    FAILED\n");
  return NumErrs == 0;
}

int main(void) {
  queue Q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<info::device::name>() << "\n";

  bool Pass = true;

  Pass &= test<int8_t, 8>(Q);
  Pass &= test<int8_t, 16>(Q);
  Pass &= test<int8_t, 32>(Q);

  Pass &= test<int16_t, 8>(Q);
  Pass &= test<int16_t, 16>(Q);
  Pass &= test<int16_t, 32>(Q);

  Pass &= test<int32_t, 8>(Q);
  Pass &= test<int32_t, 16>(Q);
  Pass &= test<int32_t, 32>(Q);

  Pass &= test<half, 8>(Q);
  Pass &= test<half, 16>(Q);
  Pass &= test<half, 32>(Q);

  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
