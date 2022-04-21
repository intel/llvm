// RUN: %clang -fsycl -O0 -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics

#include <CL/sycl.hpp>

int main() {
  cl::sycl::vec<cl::sycl::half, 1> V(1.0);
  cl::sycl::vec<cl::sycl::half, 1> V2 = V.template convert<cl::sycl::half>();

  return 0;
}
