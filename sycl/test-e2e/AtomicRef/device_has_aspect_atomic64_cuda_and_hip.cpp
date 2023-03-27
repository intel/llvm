// REQUIRES: cuda || hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
  queue Queue;
  device Dev = Queue.get_device();
  // cout in order to ensure that the query hasn't been optimized out
  std::cout << Dev.has(aspect::atomic64) << std::endl;
  return 0;
}
