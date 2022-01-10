// This test checks presence of macros for available extensions.
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
// REQUIRES: hip_be
#include <CL/sycl.hpp>
#include <iostream>
int main() {
#if SYCL_EXT_ONEAPI_BACKEND_HIP == 1
  std::cout << "SYCL_EXT_ONEAPI_BACKEND_HIP=1" << std::endl;
#else
  std::cerr << "SYCL_EXT_ONEAPI_BACKEND_HIP!=1" << std::endl;
  exit(1);
#endif
  exit(0);
}
