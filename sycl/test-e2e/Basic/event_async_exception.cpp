// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==---- event_async_exception.cpp - Test for event async exceptions -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

// This test checks that if there is a submit failure, the asynchronous
// exception is associated with the returned event.

using namespace sycl;

class KernelName;

int main() {
  auto asyncHandler = [](exception_list el) {
    for (auto &e : el) {
      std::rethrow_exception(e);
    }
  };

  queue q(asyncHandler);

  try {
    // Check that submitting a CG with no kernel or memory operation doesn't
    // produce an async exception
    event e = q.submit([&](handler &cgh) {});

    e.wait_and_throw();
    return 0;
  } catch (exception e) {
    return 1;
  }
}
