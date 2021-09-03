// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>

using namespace sycl;

int main() {
  // expected-warning@+2 {{Specialization constats is deprecated, use SYCL 2020 specialization constants instead}}
  ext::oneapi::experimental::spec_constant<int> SC;
  (void)SC;
  return 0;
}

