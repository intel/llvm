// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

typedef union _u16_to_half {
  unsigned short u;
  sycl::half h = 0.0;
} u16_to_sycl_half;

int main() {
  u16_to_sycl_half unh;
  return 0;
}
