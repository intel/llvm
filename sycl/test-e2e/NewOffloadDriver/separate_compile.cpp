//==--------------- separate_compile.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test is copied from SeparateCompile/test.cpp
// and modified to test with the New Offloading Model.
//
// The original manual pipeline
// (llvm-link -> sycl-post-link -> llvm-spirv -> clang-offload-wrapper) is
// removed as clang-linker-wrapper does not support SPIR-V objects produced by
// the old clang-offload-wrapper, and device libraries are now auto-forwarded
// by the driver.
//
// REQUIRES: target-spir
//
// >> ---- compile src1 (produces fat object with device code)
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64 -DSYCL_DISABLE_FALLBACK_ASSERT -fno-sycl-dead-args-optimization -c %s -o a.o -Wno-sycl-strict
//
// >> ---- compile src2 (produces fat object with device code)
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64 -Wno-error=unused-command-line-argument -DSYCL_DISABLE_FALLBACK_ASSERT -DB_CPP=1 -fno-sycl-dead-args-optimization -c %s -o b.o -Wno-sycl-strict
//
// >> ---- link the full hetero app
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64 a.o b.o -o app.exe %sycl_options
//
// RUN: %{run} ./app.exe

#ifdef B_CPP
// -----------------------------------------------------------------------------
#include <iostream>
#include <sycl/detail/core.hpp>

int run_test_b(int v) {
  int arr[] = {v};
  {
    sycl::queue deviceQueue;
    sycl::buffer<int, 1> buf(arr, 1);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class kernel_b>([=]() { acc[0] *= 3; });
    });
  }
  return arr[0];
}

#else // !B_CPP

// -----------------------------------------------------------------------------
#include <iostream>
#include <sycl/detail/core.hpp>

using namespace std;

const int VAL = 10;

extern int run_test_b(int);

int run_test_a(int v) {
  int arr[] = {v};
  {
    sycl::queue deviceQueue;
    sycl::buffer<int, 1> buf(arr, 1);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task<class kernel_a>([=]() { acc[0] *= 2; });
    });
  }
  return arr[0];
}

int main(int argc, char **argv) {
  bool pass = true;

  int test_a = run_test_a(VAL);
  const int GOLD_A = 2 * VAL;

  if (test_a != GOLD_A) {
    std::cout << "FAILD test_a. Expected: " << GOLD_A << ", got: " << test_a
              << "\n";
    pass = false;
  }

  int test_b = run_test_b(VAL);
  const int GOLD_B = 3 * VAL;

  if (test_b != GOLD_B) {
    std::cout << "FAILD test_b. Expected: " << GOLD_B << ", got: " << test_b
              << "\n";
    pass = false;
  }

  return pass ? 0 : 1;
}
#endif // !B_CPP
