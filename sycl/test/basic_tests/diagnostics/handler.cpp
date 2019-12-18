// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out | FileCheck %s
//==------------------- handler.cpp ----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl;

int main() {

  bool Failed = false;

  sycl::queue Queue([](sycl::exception_list ExceptionList) {
    if (ExceptionList.size() != 1) {
      std::cerr << "Should be one exception in exception list" << std::endl;
      std::abort();
    }
    std::rethrow_exception(*ExceptionList.begin());
  });

  try {
    Queue.submit([&](sycl::handler &CGH) {
      CGH.single_task<class Dummy1>([]() {});
      CGH.single_task<class Dummy2>([]() {});
    });
    Queue.throw_asynchronous();
  } catch (sycl::exception &E) {
    // CHECK: Attempt to set multiple actions for the command group
    std::cout << E.what() << std::endl;
  }
}
