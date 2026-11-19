//==--------- build-log.cpp - Test log message from failed build -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This test checks that a kernel containing invalid assembly on GPUs or
/// calling a nonexistent symbol on CPUs triggers a runtime build exception
/// containing a build log.
/// For the NVIDIA version, see: build-log-nvidia.cpp
///
//===----------------------------------------------------------------------===//

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// UNSUPPORTED: target-nvidia
// UNSUPPORTED-INTENDED: See special test file for NVIDIA:
// build-log-nvidia.cpp

// RUN: %{build} -DGPU -o %t_gpu.out
// RUN: %{build} -o %t.out

// RUN: %{run} %if gpu %{ %t_gpu.out %} %else %{ %t.out %} > %t.log 2>&1
// RUN: %{run-aux} FileCheck %s --check-prefix %if gpu %{CHECK-EXPECTED-ERROR-GPU%} %else %{CHECK-EXPECTED-ERROR-CPU%} --input-file %t.log

// CHECK-EXPECTED-ERROR-GPU: error: backend compiler failed build
// CHECK-EXPECTED-ERROR-CPU: {{Failed to build device program|error: backend compiler failed build}}

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>

SYCL_EXTERNAL
void symbol_that_does_not_exist();

void test() {
#ifdef GPU
  auto DeviceSelector = sycl::gpu_selector_v;
#else  // GPU
  auto DeviceSelector = sycl::cpu_selector_v;
#endif // GPU
  sycl::queue Queue(DeviceSelector);

  // Submitting this kernel should result in an runtime
  // exception with error code `sycl::errc::build`
  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
#ifdef GPU
    asm volatile("undefined\n");
#else  // GPU
    symbol_that_does_not_exist();
#endif // GPU
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
