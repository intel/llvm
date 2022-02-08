//==---- simd_binop_integer_promotion.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to outdated memory intrinsic
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that ESIMD binary operation APIs honor integer promotion
// rules. E.g.:
//   simd <short, 32> a;
//   simd <short, 32> b;
//   a + b; // yield simd <int, 32>

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>
#include <limits>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <typename T> struct KernelName;

// The main test routine.
template <typename T> bool test(queue q) {
  constexpr unsigned VL = 16;

  T A[VL];
  T B[VL];
  using T_promoted = decltype(T{} + T{});
  T_promoted C[VL];

  std::cout << "Testing " << typeid(T).name() << " + " << typeid(T).name()
            << " => " << typeid(T_promoted).name() << " ...\n";

  for (unsigned i = 0; i < VL; ++i) {
    A[i] = i;
    B[i] = 1;
  }
  T maxNum = std::numeric_limits<T>::max();
  // test overflow in one of the lanes
  A[VL / 2] = maxNum;
  B[VL / 2] = maxNum;

  try {
    buffer<T, 1> bufA(A, range<1>(VL));
    buffer<T, 1> bufB(B, range<1>(VL));
    buffer<T_promoted, 1> bufC(C, range<1>(VL));
    range<1> glob_range{1};

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufA.template get_access<access::mode::read>(cgh);
      auto PB = bufB.template get_access<access::mode::read>(cgh);
      auto PC = bufC.template get_access<access::mode::write>(cgh);
      cgh.parallel_for<KernelName<T>>(
          glob_range, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::experimental::esimd;
            unsigned int offset = i * VL * sizeof(T);
            simd<T, VL> va;
            va.copy_from(PA, offset);
            simd<T, VL> vb;
            vb.copy_from(PB, offset);
            auto vc = va + vb;
            unsigned int offsetC = i * VL * sizeof(T_promoted);
            vc.copy_to(PC, offsetC);
          });
    });
    q.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false; // not success
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < VL; ++i) {
    T_promoted gold = (T_promoted)(A[i] + B[i]);
    T_promoted val = C[i];

    if (val != gold) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ": " << val << " != " << gold
                  << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(VL - err_cnt) / (float)VL) * 100.0f
              << "% (" << (VL - err_cnt) << "/" << VL << ")\n";
  }

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<unsigned short>(q);
  passed &= test<short>(q);
  passed &= test<unsigned char>(q);
  passed &= test<char>(q);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
