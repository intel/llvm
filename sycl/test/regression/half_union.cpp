// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

typedef union _u16_to_half {
  unsigned short u;
  sycl::half h;
} u16_to_sycl_half;

int main() {
  u16_to_sycl_half unh;
  return 0;
}
