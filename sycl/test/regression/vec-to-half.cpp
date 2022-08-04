// RUN: %clang -fsycl -O0 -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  sycl::vec<sycl::half, 1> V(1.0);
  sycl::vec<sycl::half, 1> V2 = V.template convert<sycl::half>();

  return 0;
}
