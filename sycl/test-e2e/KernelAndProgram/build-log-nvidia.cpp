//==------ build-log-nvidia.cpp - Test log message from failed build
//--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This test checks that compiling a kernel containing invalid assembly
/// in its body with CUDA backend produces a ptxas fatal error at compile time,
/// unlike other backends that generate runtime build error.
/// For the other backends, see: build-log.cpp.
///
//===----------------------------------------------------------------------===//

// REQUIRES: target-nvidia

// RUN: not %{build} -o %t.exe > %t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK-PTXAS-ERROR --input-file %t.out

// CHECK-PTXAS-ERROR: ptxas fatal

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>

void test() {
  sycl::queue Queue(sycl::gpu_selector_v);

  // Submitting this kernel should result in compilation error with error code
  // ptxas fatal
  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile("undefined\n");
#endif // __SYCL_DEVICE_ONLY__
  };

  try {
    Queue.single_task(Kernel).wait();
  } catch (const sycl::exception &e) {
    std::cerr << e.what() << std::endl;
    assert(e.code() == sycl::errc::build &&
           "Caught exception was not a compilation error");

  } catch (...) {
    assert(false && "Caught exception was not a compilation error");
  }
}

int main() {
  test();

  return 0;
}
