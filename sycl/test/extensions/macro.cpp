// This test checks presence of macros for available extensions.
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>
int main() {
#if SYCL_BACKEND_OPENCL == 1
  std::cout << "SYCL_BACKEND_OPENCL=1" << std::endl;
#else
  std::cerr << "SYCL_BACKEND_OPENCL!=1" << std::endl;
  exit(1);
#endif
#if SYCL_EXT_ONEAPI_SUB_GROUP_MASK == 1
  std::cout << "SYCL_EXT_ONEAPI_SUB_GROUP_MASK=1" << std::endl;
#else
  std::cerr << "SYCL_EXT_ONEAPI_SUB_GROUP_MASK!=1" << std::endl;
  exit(1);
#endif
#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO == 1
  std::cout << "SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO=1" << std::endl;
#else
  std::cerr << "SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO!=1" << std::endl;
  exit(1);
#endif
  exit(0);
}
