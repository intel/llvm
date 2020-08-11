// >> ---- device compilation
// RUN: %clangxx -std=c++14 -fsycl-device-only -Xclang -fsycl-int-header=sycl_ihdr_a.h %s -c -o a_kernel.bc -I %sycl_include
//
// >> ---- host compilation: no cpp version mismatch
// RUN: %clangxx -std=c++14 -include sycl_ihdr_a.h -c %s -I %sycl_include
//
// >> ---- code generation checks
// RUN: FileCheck -input-file=sycl_ihdr_a.h %s
// CHECK: #define STD_CPP_VERSION 
// CHECK-NEXT: #if __cplusplus != STD_CPP_VERSION
// CHECK-NEXT: #error "C++ version for host compilation does not match C++ version used for device compilation"
// CHECK-NEXT: #endif

//==----------- cpp_version_mismatch_test_2.cpp - SYCL separate compilation cpp version mismatch test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// -----------------------------------------------------------------------------
#include <CL/sycl.hpp>
#include <iostream>

using namespace std;

const int VAL = 10;

// This tests uses a simple example with kernel creation 
// to help exercise integration header file generation
// and c++ version mismatch diagnostics generation
// In this case the compiler versions are the same

int run_test_a(int v) {
  int arr[] = {v};
  {
    cl::sycl::queue deviceQueue;
    cl::sycl::buffer<int, 1> buf(arr, 1);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
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

  if (pass) {
    std::cout << "pass\n";
  }
  return pass ? 0 : 1;
}
