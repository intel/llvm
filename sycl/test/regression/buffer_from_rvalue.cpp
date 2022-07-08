// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out
// expected-no-diagnostics

#include <CL/sycl.hpp>

int main() {
  std::vector<int> v(1);
  std::move_iterator it1(v.begin());
  std::move_iterator it2(v.end());
  cl::sycl::buffer b1(it1, it2);
  return 0;
}
