//==---------------- esimd_merge.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This is a functional test for esimd::merge free functions, as well as
// two-input version of the simd_obj_impl::merge.

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel;
using namespace cl::sycl;

template <class T> void prn(T *arr, int size, const char *title) {
  std::cout << title << ": ";
  for (int i = 0; i < size; ++i) {
    std::cout << " " << arr[i];
  }
  std::cout << "\n";
}

int main(void) {
  constexpr unsigned NUM_THREADS = 2;
  constexpr unsigned VL = 16;
  constexpr unsigned FACTOR = 2;
  constexpr unsigned SUB_VL = VL / FACTOR / FACTOR;
  constexpr unsigned Size = VL * NUM_THREADS;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int *A = malloc_shared<int>(Size, q);
  int *B = malloc_shared<int>(Size, q);
  using MaskElT = typename simd_mask<1>::element_type;
  MaskElT *M = malloc_shared<MaskElT>(Size, q);
  int *C = malloc_shared<int>(Size, q);
  int *C1 = malloc_shared<int>(Size, q);
  constexpr int VAL0 = 0;
  constexpr int VAL1 = 1;
  constexpr int VAL2 = 3;

  for (int i = 0; i < Size; ++i) {
    A[i] = i % 2 + VAL1; // 1212 ...
    B[i] = i % 2 + VAL2; // 3434 ...
    // mask out first half of sub-vector, alternating '1' and '2' as 'enabled'
    // bit representation in a mask element:
    M[i] = (i % SUB_VL) >= (SUB_VL / 2) ? (i % SUB_VL - 1) : 0; // 00120012 ...
    C[i] = VAL0;
    C1[i] = VAL0;
  }

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(NUM_THREADS, [=](id<1> i) SYCL_ESIMD_KERNEL {
        simd<int, VL> va(A + i * VL);
        simd<int, VL> vb(B + i * VL);
        simd_mask<SUB_VL> m(M + i * VL);
        // va:         1212121212121212
        // vb:         3434343434343434
        // m:          0012001200120012
        // va.sel.sel: 1111
        // vb.sel.sel: 4444
        // vc:         4411
        simd<int, SUB_VL> vc =
            esimd::merge(va.select<SUB_VL * 2, 2>(0).select<SUB_VL, 1>(1),
                         vb.select<SUB_VL * 1, 2>(1).select<SUB_VL, 1>(0), m);
        vc.copy_to(C + i * VL);

        // also check that
        //   vc = esimd::merge(a, b, m)
        // is equivalent to
        //   vc.merge(a, b, m)
        simd<int, SUB_VL> vc1;
        vc1.merge(va.select<SUB_VL * 2, 2>(0).select<SUB_VL, 1>(1).read(),
                  vb.select<SUB_VL * 1, 2>(1).select<SUB_VL, 1>(0).read(), m);
        vc1.copy_to(C1 + i * VL);
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    sycl::free(C1, q);
    sycl::free(M, q);
    return 1;
  }

  int err_cnt = 0;

  prn(A, Size, "A");
  prn(B, Size, "B");
  prn(M, Size, "M");
  prn(C, Size, "C");

  for (int i = 0; i < Size; ++i) {
    int j = i % VL;
    int gold =
        j >= SUB_VL ? VAL0 : ((j % SUB_VL) >= (SUB_VL / 2) ? VAL1 : VAL2 + 1);

    if (C[i] != gold) {
      if (++err_cnt < 10) {
        std::cout << "(esimd::merge) failed at index " << i << ", " << C[i]
                  << " != " << gold << " (gold)\n";
      }
    }
    if (C1[i] != gold) {
      if (++err_cnt < 10) {
        std::cout << "(simd::merge) failed at index " << i << ", " << C1[i]
                  << " != " << gold << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
  sycl::free(C1, q);
  sycl::free(M, q);
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
