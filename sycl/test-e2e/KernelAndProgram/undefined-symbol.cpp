// for CUDA and HIP the failure happens at compile time, not during runtime
// UNSUPPORTED: cuda || hip

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
//==--- undefined-symbol.cpp - Error handling for undefined device symbol --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
SYCL_EXTERNAL
void symbol_that_does_not_exist();

void test() {
  sycl::queue Queue;

  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
    // This is not guaranteed by the SYCL specification, but DPC++ currently
    // does not diagnose an error at compilation/link time if a kernel
    // references an undefined symbol from within code that is protected by
    // __SYCL_DEVICE_ONLY__.  As a result, this error is delayed and diagnosed
    // at runtime when the kernel is submitted to a device.
    //
    // This test is "unsupported" on the host device because the kernel does
    // not actually contain an undefined reference in that case.
    symbol_that_does_not_exist();
#endif // __SYCL_DEVICE_ONLY__
  };

  try {
    Queue.submit(
        [&](sycl::handler &CGH) { CGH.single_task<class SingleTask>(Kernel); });
    assert(false && "Expected error submitting kernel");
  } catch (const sycl::exception &e) {
    assert((e.code() == sycl::errc::build) && "Wrong error code");

    // Error message should mention name of undefined symbol.
    std::string Msg(e.what());
    assert(Msg.find("symbol_that_does_not_exist") != std::string::npos);
  } catch (...) {
    assert(false && "Expected sycl::exception");
  }
}

int main() {
  test();

  return 0;
}
