// RUN: %clangxx -std=gnu++11 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out

// UNSUPPORTED: system-windows

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  std::cout << "Passed" << std::endl;
  return 0;
}
