// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main() {
  sycl::vec<float, 1> A{1}, B{2}, C{3};
  sycl::vec<float, 1> res = sycl::mad(A, B, C);
  assert(res.x() - 5.0f < 1e-5);

  res = sycl::clamp(A, B, C);
  assert(res.x() - 2.0f < 1e-5);

  float scalarRes = sycl::length(A);
  assert(scalarRes - 1.0f < 1e-5);
}
