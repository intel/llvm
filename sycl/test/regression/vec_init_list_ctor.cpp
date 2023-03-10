// RUN: %clang -fsycl -O0 -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// Regression test checking that the vector ctor taking an initializer list
// doesn't cause warnings or errors.

#include <sycl/sycl.hpp>

int main() {
  sycl::vec<int, 2> V({1, 2});
  return 0;
}
