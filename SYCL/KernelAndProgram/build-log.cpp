// XFAIL: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--- build-log.cpp - Test log message from faild build ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include <CL/sycl.hpp>

SYCL_EXTERNAL
void symbol_that_does_not_exist();

void test() {
  cl::sycl::queue Queue;

  // Submitting this kernel should result in a compile_program_error exception
  // with a message indicating that "symbol_that_does_not_exist" is undefined.
  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
    symbol_that_does_not_exist();
#endif
  };

  std::string Msg;
  int Result;

  try {
    Queue.submit([&](cl::sycl::handler &CGH) {
      CGH.single_task<class SingleTask>(Kernel);
    });
    assert(false && "There must be compilation error");
  } catch (const cl::sycl::compile_program_error &e) {
    std::string Msg(e.what());
    assert(Msg.find("symbol_that_does_not_exist") != std::string::npos);
  } catch (...) {
    assert(false && "There must be cl::sycl::compile_program_error");
  }
}

int main() {
  test();

  return 0;
}
