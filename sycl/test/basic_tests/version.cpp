// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>
#include <iostream>

int main() {
#ifndef __LIBSYCL_MAJOR_VERSION
  std::cerr << "__LIBSYCL_MAJOR_VERSION was not found\n";
  return -1;
#endif
#ifndef __LIBSYCL_MINOR_VERSION
  std::cerr << "__LIBSYCL_MINOR_VERSION was not found\n";
  return -1;
#endif
#ifndef __LIBSYCL_PATCH_VERSION
  std::cerr << "__LIBSYCL_PATCH_VERSION was not found\n";
  return -1;
#endif

  return 0;
}
