//==--- lsc_usm_atomic_cachehint.cpp - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <stdlib.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

class Test;

#define DTYPE float

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

ESIMD_INLINE void atomic_add_float(DTYPE *sA, simd_mask<16> M) {
  simd<uint32_t, 16> offsets(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  simd<float, 16> mat({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.5, 0.5, 0.5, 0.5});
  lsc_atomic_update<atomic_op::fadd, float, 16, lsc_data_size::default_size,
                    cache_hint::uncached, cache_hint::write_back>(
      (float *)sA, offsets * sizeof(float), mat, M);
}

int main(void) {
  constexpr unsigned Size = 256;
  constexpr unsigned VL = 16;
  constexpr size_t LocalRange = 4;
  constexpr size_t GlobalRange = 64;

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctxt = q.get_context();

  DTYPE *A = malloc_shared<DTYPE>(VL, q);
  DTYPE *B = malloc_shared<DTYPE>(VL, q);
  DTYPE *C = malloc_shared<DTYPE>(VL, q);
  DTYPE *D = malloc_shared<DTYPE>(VL, q);

  for (unsigned i = 0; i < VL; ++i) {
    A[i] = 0;
    B[i] = GlobalRange * 0.5; // expect changes in all elements
    C[i] = 0;
    D[i] = (i & 1) ? B[i] : 0; // expect changes in elements with odd indices
  }

  nd_range<1> Range(range<1>{GlobalRange}, range<1>{LocalRange});

  sycl::ext::oneapi::experimental::properties prop{
      sycl::ext::intel::experimental::grf_size<256>};
  try {
    q.submit([&](handler &cgh) {
       cgh.parallel_for<Test>(
           Range, prop, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
             atomic_add_float(A, 1);
             simd_mask<16> M({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
             atomic_add_float(C, M);
           });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(B, q);
    free(C, q);
    free(D, q);
    return 1;
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < VL; ++i) {
    if (A[i] != B[i]) {
      if (++err_cnt < 10)
        std::cerr << "A == B failed at " << i << ": " << A[i] << " != " << B[i]
                  << "\n";
    }
    if (C[i] != D[i]) {
      if (++err_cnt < 10)
        std::cerr << "C == D failed at " << i << ": " << C[i] << " != " << D[i]
                  << "\n";
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(VL - err_cnt) / (float)VL) * 100.0f
              << "% (" << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  free(A, q);
  free(B, q);
  free(C, q);
  free(D, q);

  return err_cnt > 0 ? 1 : 0;
}
