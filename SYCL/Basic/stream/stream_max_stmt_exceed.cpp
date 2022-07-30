// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_ON_LINUX_PLACEHOLDER %t.out %GPU_CHECK_ON_LINUX_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

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
