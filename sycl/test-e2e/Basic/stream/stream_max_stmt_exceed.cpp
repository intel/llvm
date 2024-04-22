// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %if !gpu || linux %{ | FileCheck %s %}

#include <cassert>
#include <iostream>

#include <sycl/detail/core.hpp>

#include <sycl/stream.hpp>

int main() {
  sycl::queue Queue;
  try {
    Queue.submit([&](sycl::handler &cgh) {
      sycl::stream Out(100, 65536, cgh);
      cgh.single_task<class test_max_stmt_exceed>(
          [=]() { Out << "Hello world!" << sycl::endl; });
    });
    Queue.wait();
  } catch (sycl::exception &ExpectedException) {
    // CHECK: Maximum statement size exceeds limit of 65535 bytes
    std::cout << ExpectedException.what() << std::endl;
  }
  return 0;
}
