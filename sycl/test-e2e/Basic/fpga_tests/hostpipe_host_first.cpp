//==----  hostpipe_host_first.cpp - Verify host access before kernel  ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: accelerator
// RUN: %clangxx -fsycl %s -o %t.out -fsycl-targets=%sycl_triple
// RUN: %ACC_RUN_PLACEHOLDER %t.out


#include <sycl/sycl.hpp>
#include <iostream>

#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/experimental/pipe_properties.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

using default_pipe_properties =
    decltype(sycl::ext::oneapi::experimental::properties());

// Classes used to name the kernels
class TestTask;
class H2DPipeID;
class D2HPipeID;

using H2DPipe = sycl::ext::intel::experimental::pipe<H2DPipeID, int, 10, default_pipe_properties>;
using D2HPipe = sycl::ext::intel::experimental::pipe<D2HPipeID, int, 10, default_pipe_properties>;

struct BasicKernel {
  void operator()() const { 
    auto a = H2DPipe::read();
    D2HPipe::write(a+1);
  }
};

int main() {
  default_selector selector;
  queue q(selector);

  H2DPipe::write(q, 0xdeadbeef);  

  q.submit([&](handler &h) {
    h.single_task<TestTask>(BasicKernel{});
  }).wait();

  auto b = D2HPipe::read(q);
  std::cout << "READ BACK: " << b << std::endl;

  if (b == 0xdeadbeef + 1) {
    std::cout << "Test passed" << std::endl;
    return 0;
  } else {
    std::cout << "Test failed" << std::endl;
    return 1;
  }
}

