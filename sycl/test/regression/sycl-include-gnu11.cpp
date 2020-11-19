// RUN: %clangxx -std=gnu++11 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: system-windows

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  std::cout << "Passed" << std::endl;
  return 0;
}
