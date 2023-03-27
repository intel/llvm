//===------ noinline_call_recursive.cpp  - DPC++ ESIMD on-device test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// Recursion is not supported in ESIMD (intel/llvm PR#3390)
// REQUIRES: TEMPORARY_DISBLED
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env IGC_FunctionControl=3 IGC_ForceInlineStackCallWithImplArg=1 %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that ESIMD kernels support recursive call of noinline
// functions.

#include "esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

class KernelID;

ESIMD_NOINLINE unsigned add(unsigned A, unsigned B, unsigned C) {
  if (B == 0)
    return A;

  return (B % C == 0) ? add(A + 1, B - 1, C) : add(A - C, B - 2, C);
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  unsigned result = 0;
  unsigned *output = &result;

  unsigned in1 = 233;
  unsigned in2 = 21;
  unsigned in3 = 3;

  try {
    buffer<unsigned, 1> buf(output, range<1>(1));

    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<KernelID>(sycl::range<1>{1},
                                 [=](id<1> i) SYCL_ESIMD_KERNEL {
                                   using namespace sycl::ext::intel::esimd;

                                   auto res = add(in1, in2, in3);

                                   scalar_store(acc, 0, res);
                                 });
    });
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return e.get_cl_code();
  }

  int etalon = in1;
  while (in2 > 0) {
    if (in2 % in3 == 0) {
      etalon += 1;
      in2 -= 1;
    } else {
      etalon -= in3;
      in2 -= 2;
    }
  }

  if (result != etalon) {
    std::cout << "Failed: " << result << std::endl;
    return 1;
  }

  return 0;
}
