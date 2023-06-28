// RUN: %clangxx -fsycl -DNDEBUG %s -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

#include <algorithm>
#include <limits>

int main() {
  double f1 = 1.0f;
  double f2 = std::numeric_limits<double>::quiet_NaN();

  assert(((std::max(f1, f2) == sycl::maximum{}(f1, f2)) &&
          "sycl::maximum result is wrong"));
  assert(((std::isnan((std::max(f2, f1)))) &&
          (std::isnan(sycl::maximum{}(f2, f1)))) &&
         "sycl::maximum result is wrong");
  assert(((std::min(f1, f2) == sycl::minimum{}(f1, f2)) &&
          "sycl::minimum result is wrong"));
  assert(((std::isnan((std::min(f2, f1)))) &&
          (std::isnan(sycl::minimum{}(f2, f1)))) &&
         "sycl::minimum result is wrong");

  return 0;
}
