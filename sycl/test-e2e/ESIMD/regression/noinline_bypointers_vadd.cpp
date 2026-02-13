//==------ noinline_bypointers_vadd.cpp  - DPC++ ESIMD on-device test ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

using ptr = float *;
static inline constexpr unsigned VL = 32;

SYCL_EXTERNAL ESIMD_NOINLINE void do_add(ptr A, float *B,
                                         ptr C) SYCL_ESIMD_FUNCTION {
  simd<float, VL> va;
  va.copy_from(A);
  simd<float, VL> vb;
  vb.copy_from(B);
  simd<float, VL> vc = va + vb;
  vc.copy_to(C);
}

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 8;

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  auto ctxt = q.get_context();
  float *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *B =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  // We need that many workitems. Each processes VL elements of data.
  sycl::range<1> GlobalRange{Size / VL};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            int i = ndi.get_global_id(0);
            do_add(ptr{A + i * VL}, B + i * VL, ptr{C + i * VL});
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    free(A, ctxt);
    free(B, ctxt);
    free(C, ctxt);

    return e.code().value();
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << "\n";
      }
    }
  }

  free(A, ctxt);
  free(B, ctxt);
  free(C, ctxt);

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
    std::cout << "FAILED\n";
    return 1;
  }

  std::cout << "Passed\n";
  return 0;
}
