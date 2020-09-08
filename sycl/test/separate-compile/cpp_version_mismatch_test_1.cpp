// >> ---- device compilation
// RUN: %clangxx -std=c++14 -fsycl-device-only -Xclang -fsycl-int-header=sycl_ihdr_a.h %s -c -I %sycl_include

// >> ---- host compilation: cpp version mismatch
// RUN: not %clangxx -std=c++11 -include sycl_ihdr_a.h -c %s -I %sycl_include 2>&1 | FileCheck %s

// >> ---- diagnostics correctness check
// CHECK: C++ version for host compilation does not match C++ version used for device compilation

//==----------- cpp_version_mismatch_test_1.cpp - SYCL separate compilation cpp version mismatch test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// -----------------------------------------------------------------------------
#include <CL/sycl.hpp>

// This tests uses a simple example with kernel creation
// to help exercise integration header file generation
// and c++ version mismatch diagnostics generation
// In this case the compiler versions are different

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  // Run empty kernel
  sycl::queue deviceQueue;
  deviceQueue.submit([&](sycl::handler& cgh) {
    cgh.single_task<class kernel_a>([=]() { });
  });

  return 0;
}
