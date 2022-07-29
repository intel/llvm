//===------ noinline_call_from_func.cpp  - DPC++ ESIMD on-device test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// Test currently timeouts on Windows Level Zero and OpenCL
// UNSUPPORTED: windows
// Temporarily disabled due to flaky behavior
// REQUIRES: TEMPORARY_DISABLED
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env IGC_FunctionControl=3 IGC_ForceInlineStackCallWithImplArg=1 %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda || hip
//
// The test checks that ESIMD kernels support call of noinline function from
// within other functions.

#include "esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

class KernelID;

ESIMD_NOINLINE int add(int A, int B) { return A + B; }

template <typename AccTy> ESIMD_NOINLINE void test(AccTy acc, int A, int B) {
  using namespace sycl::ext::intel::esimd;

  auto res = add(A, B);

  scalar_store(acc, 0, res);
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int result = 0;
  int *output = &result;

  int in1 = 100;
  int in2 = 233;

  try {
    buffer<int, 1> buf(output, range<1>(1));

    q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<KernelID>(
          sycl::range<1>{1},
          [=](id<1> i) SYCL_ESIMD_KERNEL { test(acc, in1, in2); });
    });
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return e.get_cl_code();
  }

  if (result != (in1 + in2)) {
    std::cout << "Failed: with result: " << result << std::endl;
    return 1;
  }

  std::cout << "passed" << std::endl;
  return 0;
}
