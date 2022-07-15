// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <sycl/sycl.hpp>

typedef union _u16_to_half {
  unsigned short u;
  cl::sycl::half h;
} u16_to_sycl_half;

int main() {
  u16_to_sycl_half unh;
  return 0;
}
