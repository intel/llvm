// This test checks presence of macros for available extensions.
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
// REQUIRES: esimd_emulator_be
#include <CL/sycl.hpp>
#include <iostream>
int main() {
#if SYCL_EXT_INTEL_BACKEND_ESIMD_EMULATOR == 1
  std::cout << "SYCL_EXT_INTEL_BACKEND_ESIMD_EMULATOR=1" << std::endl;
#else
  std::cerr << "SYCL_EXT_INTEL_BACKEND_ESIMD_EMULATOR!=1" << std::endl;
  exit(1);
#endif
  exit(0);
}
