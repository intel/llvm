// RUN: %clangxx %s -o %t.out -lOpenCL -lsycl
// RUN: %t.out

// Test that vector with 'unsigned long long' elements has enough bits to store
// value.

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>

int main(void) {
  unsigned long long ref = 1ull - 2ull;
  auto vec = cl::sycl::vec<unsigned long long, 1>(ref);
  unsigned long long val = vec.template swizzle<cl::sycl::elem::s0>();

  assert(val == ref);
  return 0;
}
