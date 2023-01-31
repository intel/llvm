//==---- svm_gather_scatter.cpp  - DPC++ ESIMD svm gather/scatter test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu && !gpu-intel-pvc
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl-device-code-split=per_kernel -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Regression test for SVM gather/scatter API.

#include "../esimd_test_utils.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <sycl/builtins_esimd.hpp>
#include <sycl/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;
using bfloat16 = sycl::ext::oneapi::bfloat16;
using tfloat32 = sycl::ext::intel::experimental::esimd::tfloat32;

#ifdef USE_64_BIT_OFFSET
typedef uint64_t Toffset;
#else
typedef uint32_t Toffset;
#endif

template <typename T, int N> bool test(queue &Q) {
  std::cout << "  Running " << esimd_test::type_name<T>() << " test, N=" << N
            << "...\n";

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
#ifndef USE_SCALAR_OFFSET
         simd<Toffset, N> Offsets(0u, sizeof(T));
#else
             Toffset Offsets = 0;
#endif
         scatter<T, N>(Dst, Offsets, gather<T, N>(Src, Offsets));
       });
     }).wait();
  } catch (sycl::exception const &E) {
    std::cout << "ERROR. SYCL exception caught: " << E.what() << std::endl;
    return false;
  }

  unsigned NumErrs = 0;
  for (int I = 0; I < N; ++I)
#ifndef USE_SCALAR_OFFSET
    if (Dst[I] != Src[I])
#else
    if ((Dst[I] != Src[I] && I == 0) || (I != 0 && Dst[I] != 0))
#endif
      if (++NumErrs <= 10)
        std::cout << "failed at " << I << ": " << Dst[I]
                  << " (Dst) != " << Src[I] << " (Src)\n";

  std::cout << (NumErrs == 0 ? "    Passed\n" : "    FAILED\n");
  return NumErrs == 0;
}

int main(void) {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto Dev = Q.get_device();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>()
            << "\n";

  bool Pass = true;

  Pass &= test<int8_t, 1>(Q);
#ifndef USE_SCALAR_OFFSET
  Pass &= test<int8_t, 2>(Q);
  Pass &= test<int8_t, 4>(Q);
  Pass &= test<int8_t, 8>(Q);
  Pass &= test<int8_t, 16>(Q);
  Pass &= test<int8_t, 32>(Q);
#endif

  Pass &= test<int16_t, 1>(Q);
#ifndef USE_SCALAR_OFFSET
  Pass &= test<int16_t, 2>(Q);
  Pass &= test<int16_t, 4>(Q);
  Pass &= test<int16_t, 8>(Q);
  Pass &= test<int16_t, 16>(Q);
  Pass &= test<int16_t, 32>(Q);
#endif

  Pass &= test<int32_t, 1>(Q);
#ifndef USE_SCALAR_OFFSET
  Pass &= test<int32_t, 2>(Q);
  Pass &= test<int32_t, 4>(Q);
  Pass &= test<int32_t, 8>(Q);
  Pass &= test<int32_t, 16>(Q);
  Pass &= test<int32_t, 32>(Q);
#endif
  if (Dev.has(aspect::fp16)) {
    Pass &= test<half, 1>(Q);
#ifndef USE_SCALAR_OFFSET
    Pass &= test<half, 2>(Q);
    Pass &= test<half, 4>(Q);
    Pass &= test<half, 8>(Q);
    Pass &= test<half, 16>(Q);
    Pass &= test<half, 32>(Q);
#endif
  }

  Pass &= test<bfloat16, 1>(Q);
#ifndef USE_SCALAR_OFFSET
  Pass &= test<bfloat16, 2>(Q);
  Pass &= test<bfloat16, 4>(Q);
  Pass &= test<bfloat16, 8>(Q);
  Pass &= test<bfloat16, 16>(Q);
  Pass &= test<bfloat16, 32>(Q);
#endif
#ifdef USE_TF32
  Pass &= test<tfloat32, 1>(Q);
#ifndef USE_SCALAR_OFFSET
  Pass &= test<tfloat32, 2>(Q);
  Pass &= test<tfloat32, 4>(Q);
  Pass &= test<tfloat32, 8>(Q);
  Pass &= test<tfloat32, 16>(Q);
  Pass &= test<tfloat32, 32>(Q);
#endif
#endif

  std::cout << (Pass ? "Test Passed\n" : "Test FAILED\n");
  return Pass ? 0 : 1;
}
