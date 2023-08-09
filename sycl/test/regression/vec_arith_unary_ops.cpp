// RUN: %clangxx -fsycl -fsyntax-only %s

// Tests that the arithmetic unary operators - and + on vectors that do not use
// native OpenCL vector implementations compile.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  Q.single_task([=]() {
    sycl::vec<long, 16> V1{32};
    auto V2 = -V1;
    auto V3 = +V2;
  });
  return 0;
}
