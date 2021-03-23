// RUN: %clangxx -std=gnu++17 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// UNSUPPORTED: system-windows

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  std::cout << "Passed" << std::endl;
  return 0;
}
