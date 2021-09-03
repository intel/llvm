// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>

using namespace sycl;

int main() {
  // expected-warning@+1 {{Specialization constats extension is deprecated, use SYCL 2020 specialization constants instead}}
  ext::oneapi::experimental::spec_constant<int> SC;
  (void)SC;
  return 0;
}

