// for CUDA and HIP the failure happens at compile time, not during runtime
// UNSUPPORTED: cuda || hip || ze_debug
// TODO: rewrite this into a unit-test

// RUN: %{build} -DGPU -o %t_gpu.out
// RUN: %{build} -o %t.out
// RUN: %{run} %if gpu %{ %t_gpu.out %} %else %{ %t.out %}
//
//==--- build-log.cpp - Test log message from faild build ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include <iostream>
#include <sycl/detail/core.hpp>
SYCL_EXTERNAL
void symbol_that_does_not_exist();

void test() {
  sycl::queue Queue;

  // Submitting this kernel should result in an exception with error code
  // `sycl::errc::build` and a message indicating
  // "PI_ERROR_BUILD_PROGRAM_FAILURE".
  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
#ifdef GPU
    asm volatile("undefined\n");
#else  // GPU
    symbol_that_does_not_exist();
#endif // GPU
#endif // __SYCL_DEVICE_ONLY__
  };

  std::string Msg;
  int Result;

  try {
    Queue.submit(
        [&](sycl::handler &CGH) { CGH.single_task<class SingleTask>(Kernel); });
    assert(false && "There must be compilation error");
  } catch (const sycl::exception &e) {
    std::string Msg(e.what());
    std::cerr << Msg << std::endl;
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
