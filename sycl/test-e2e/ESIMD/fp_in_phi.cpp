//==--------------- fp_in_phi.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// On Windows vector compute backend (as a part of IGC) uses llvm-7 and llvm-7
// based spirv translator. This test should start working on Windows when the
// llvm version is switched to 9.
// UNSUPPORTED: windows
// RUN: %clangxx -Xclang -fsycl-allow-func-ptr -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda || hip
// UNSUPPORTED: ze_debug-1,ze_debug4
//
// The test checks that ESIMD kernels correctly handle function pointers as
// arguments of LLVM's PHI function.

#include "esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

class KernelID;

ESIMD_NOINLINE int f1(int x) { return x + 1; }

ESIMD_NOINLINE int f2(int x) { return x + 2; }

ESIMD_NOINLINE int f3(int x) { return x + 3; }

bool test(queue q, bool flag) {
  int result = 0;
  int *output = &result;

  std::vector<int> Y = {0, 1};

  int in1 = 233;
  int in2 = 1;

  try {
    buffer<int, 1> o_buf(output, range<1>(1));
    buffer<int, 1> y_buf(Y.data(), Y.size());

    q.submit([&](handler &cgh) {
      auto o_acc = o_buf.get_access<access::mode::write>(cgh);
      auto y_acc = y_buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<KernelID>(sycl::range<1>{1},
                                 [=](id<1> i) SYCL_ESIMD_KERNEL {
                                   using namespace sycl::ext::intel::esimd;
                                   using f = int (*)(int);

                                   f a[] = {f1, f2};
                                   if (flag) {
                                     a[0] = f3;
                                     scalar_store(y_acc, 0, 2);
                                   }

                                   auto res = a[0](in1) + a[1](in2);

                                   scalar_store(o_acc, 0, res);
                                 });
    });
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return false;
  }

  int etalon = in1 + (flag ? 3 : 1) + in2 + 2;

  if (result != etalon) {
    std::cout << "Failed with result: " << result << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test(q, true);
  passed &= test(q, false);

  return passed ? 0 : 1;
}
