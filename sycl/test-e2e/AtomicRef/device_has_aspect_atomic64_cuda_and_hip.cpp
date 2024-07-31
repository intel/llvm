// REQUIRES: cuda || hip
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue Queue;
  device Dev = Queue.get_device();
  // cout in order to ensure that the query hasn't been optimized out
  std::cout << Dev.has(aspect::atomic64) << std::endl;
  return 0;
}
