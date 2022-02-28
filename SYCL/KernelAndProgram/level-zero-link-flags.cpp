// RUN: %clangxx -fsycl -Xsycl-target-linker=spir64 -foo %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// REQUIRES: level_zero
//==--- level-zero-link-flags.cpp - Error handling for link flags --==//
//
// The Level Zero backend does not accept any online linker options.
// This test validates that an error is raised if you attempt to pass any.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include <sycl/sycl.hpp>

class MyKernel;

void test() {
  sycl::queue Queue;
  sycl::context Context = Queue.get_context();

  auto BundleInput =
      sycl::get_kernel_bundle<MyKernel, sycl::bundle_state::input>(Context);
  auto BundleObject = sycl::compile(BundleInput);

  try {
    sycl::link(BundleObject);
    assert(false && "Expected error linking kernel bundle");
  } catch (const sycl::exception &e) {
    std::string Msg(e.what());
    assert((e.code() == sycl::errc::build) && "Wrong error code");
    assert(Msg.find("-foo") != std::string::npos);
  } catch (...) {
    assert(false && "Expected sycl::exception");
  }

  Queue.submit([&](sycl::handler &CGH) { CGH.single_task<MyKernel>([=] {}); })
      .wait();
}

int main() {
  test();

  return 0;
}
