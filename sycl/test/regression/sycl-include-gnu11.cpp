// RUN: %clangxx -std=gnu++11 -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: system-windows

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  std::cout << "Passed" << std::endl;
  return 0;
}
