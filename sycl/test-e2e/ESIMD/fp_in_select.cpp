//==--------------- fp_in_select.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: %{run} %t.out
//
// The test checks that ESIMD kernels correctly handle function pointers as
// arguments of LLVM's select function.

#include "esimd_test_utils.hpp"

class KernelID;

ESIMD_NOINLINE int add(int a, int b) { return a + b; }

ESIMD_NOINLINE int sub(int a, int b) { return a - b; }

bool test(queue q, bool flag) {
  int result = 0;
  int *output = &result;

  int in1 = 233;
  int in2 = 100;

  try {
    buffer<int, 1> buf(output, range<1>(1));

    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<KernelID>(sycl::range<1>{1},
                                 [=](id<1> i) SYCL_ESIMD_KERNEL {
                                   using namespace sycl::ext::intel::esimd;

                                   auto foo = flag ? &add : &sub;
                                   auto res = foo(in1, in2);

                                   scalar_store(acc, 0, res);
                                 });
    });
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return false;
  }

  int etalon = flag ? in1 + in2 : in1 - in2;

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
