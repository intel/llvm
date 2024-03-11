// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// Checks that sycl::nextafter with sycl::half on host correctly converts based
// on half-precision.

#include <sycl/sycl.hpp>

void check(uint16_t x, uint16_t y, uint16_t ref) {
  assert(sycl::nextafter(sycl::bit_cast<sycl::half>(x),
                         sycl::bit_cast<sycl::half>(y)) ==
         sycl::bit_cast<sycl::half>(ref));
}

int main() {
  check(0x0, 0x0, 0x0);
  check(0x1, 0x1, 0x1);
  check(0x8001, 0x8001, 0x8001);
  check(0x0, 0x1, 0x1);
  check(0x8000, 0x8001, 0x8001);
  check(0x0, 0x8001, 0x8001);
  check(0x8000, 0x1, 0x1);
  check(0x8001, 0x0, 0x0);
  check(0x1, 0x8000, 0x8000);
  check(0x8001, 0x1, 0x0);
  check(0x1, 0x8001, 0x8000);

  std::cout << "Passed!" << std::endl;
  return 0;
}
