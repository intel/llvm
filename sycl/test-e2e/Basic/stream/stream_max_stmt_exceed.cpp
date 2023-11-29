// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %if !gpu || linux %{ | FileCheck %s %}

#include <iostream>
#include <sycl/sycl.hpp>

#include <cassert>

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
