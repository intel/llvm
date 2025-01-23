// REQUIRES: accelerator
// RUN: %clangxx -fsycl -fintelfpga %s %S/kernel.cpp -I%S -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// Test checks that host pipe initialization doesn't fail if pipe is used in
// multiple translation units.

#include "mypipe.hpp"
#include <iostream>

int main() {
  sycl::queue q{sycl::ext::intel::fpga_emulator_selector_v};
  q.submit([&](sycl::handler &cgh) { cgh.single_task(KernelFunctor{}); });
  uint32_t result = KernelFunctor::my_pipe::read(q);
  q.wait();
  // CHECK: 2
  std::cout << result << std::endl;

  return 0;
}
