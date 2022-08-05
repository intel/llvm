// for CUDA and HIP the failure happens at compile time, not during runtime
// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DGPU %s -o %t_gpu.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t_gpu.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//==--- build-log.cpp - Test log message from faild build ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>
SYCL_EXTERNAL
void symbol_that_does_not_exist();

void test() {
  sycl::queue Queue;

  // Submitting this kernel should result in a compile_program_error exception
  // with a message indicating "PI_ERROR_BUILD_PROGRAM_FAILURE".
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
  } catch (const sycl::compile_program_error &e) {
    std::string Msg(e.what());
    std::cerr << Msg << std::endl;
    assert(Msg.find("PI_ERROR_BUILD_PROGRAM_FAILURE") != std::string::npos);
  } catch (...) {
    assert(false && "There must be sycl::compile_program_error");
  }
}

int main() {
  test();

  return 0;
}
