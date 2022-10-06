// RUN: %clangxx -std=gnu++17 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

// UNSUPPORTED: system-windows

#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  std::cout << "Passed" << std::endl;
  return 0;
}
