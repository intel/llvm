// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="--unknown-option" %t.out
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="--unknown-option" %t.out
// RUN: %ACC_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="--unknown-option" %t.out
//
// Unknown options are silently ignored by IGC and CUDA JIT compilers. The issue
// is under investigation.
// XFAIL: (opencl || level_zero || cuda) && gpu

//==--- build-log.cpp - Test log message from faild build ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include <CL/sycl.hpp>

void test() {
  cl::sycl::queue Queue;

  // Submitting this kernel should result in a compile_program_error exception
  // with a message indicating "Unrecognized build options".
  auto Kernel = []() {};

  std::string Msg;
  int Result;

  try {
    Queue.submit([&](cl::sycl::handler &CGH) {
      CGH.single_task<class SingleTask>(Kernel);
    });
    assert(false && "There must be compilation error");
  } catch (const cl::sycl::compile_program_error &e) {
    std::string Msg(e.what());
    std::cerr << Msg << std::endl;
    assert(Msg.find("unknown-option") != std::string::npos);
  } catch (...) {
    assert(false && "There must be cl::sycl::compile_program_error");
  }
}

int main() {
  test();

  return 0;
}
