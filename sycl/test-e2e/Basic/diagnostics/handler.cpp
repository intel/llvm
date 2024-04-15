// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s
//
// Appears to fail on HIP Nvidia because 'no device of requested type available'
// when constructing a queue with an exception_list.
// XFAIL: hip_nvidia
//==------------------- handler.cpp ----------------------------------------==//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>

int main() {
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
    assert(!"Expected exception not caught");
  } catch (sycl::exception &ExpectedException) {
    // CHECK: Attempt to set multiple actions for the command group
    std::cout << ExpectedException.what() << std::endl;
  }
}
