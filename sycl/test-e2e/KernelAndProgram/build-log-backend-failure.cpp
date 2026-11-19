//==--- build-log-backend-failure.cpp - Backend build failure log test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This test tries to launch a GPU kernel whose parameter is larger than the
/// device's max_parameter_size, resulting in backend compiler build failure
/// and an exception with the build log thrown at run time.
/// For the NVIDIA version, see: build-log-backend-failure-nvidia.cpp.
///
//===----------------------------------------------------------------------===//

// UNSUPPORTED: target-nvidia
// UNSUPPORTED-INTENDED: See special test file for nvidia:
// build-log-backend-failure-nvidia.cpp

// REQUIRES: gpu

// RUN: %{build} -o %t.exe

// RUN: %{run} %t.exe > %t.out 2>&1
// RUN: %{run-aux} FileCheck %s --check-prefix=CHECK-EXPECTED-ERROR --input-file %t.out

// CHECK-EXPECTED-ERROR: error: backend compiler failed build

#include <array>
#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>

void test() {
  using namespace sycl;

  queue q(gpu_selector_v);

  // Sized to exceed largest currently allowed kernel parameter size (CUDA -
  // 32,764 bytes) as cited in:
  // https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/
  //
  // Needs to be specified at compile time as it is used for OverflowArray's
  // size.
  constexpr size_t KernelParameterSizeOverflow = 1 + 1024 * 32;

  // In case kernel parameter limit grows in future
  // we check the current overflow value against device's max_parameter_size.
  size_t KernelParameterSizeLimit =
      q.get_device().get_info<sycl::info::device::max_parameter_size>();

  assert(KernelParameterSizeLimit < KernelParameterSizeOverflow &&
         "KernelParameterSizeOverflow must exceed the backend kernel "
         "parameter size limit; increase it in this test");

  // Stack-allocated array captured by value, can't be dynamically
  // allocated as then only the internal structure gets copied as a kernel
  // parameter.
  std::array<char, KernelParameterSizeOverflow> OverflowArray;
  OverflowArray[0] = 'a';
  OverflowArray[KernelParameterSizeOverflow - 1] = 'a';

  try {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsycl-strict"
    q.single_task([=] {
       volatile int NonDiscardValue =
           OverflowArray[0] + OverflowArray[KernelParameterSizeOverflow - 1];
       (void)NonDiscardValue;
     }).wait();
#pragma clang diagnostic pop
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
