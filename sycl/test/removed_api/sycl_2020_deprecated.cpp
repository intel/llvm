// RUN: %clangxx %fsycl-host-only -fsyntax-only -sycl-std=2020 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>

int main() {
  // expected-error@+1{{no member named 'max_constant_buffer_size' in 'sycl::info::device'}}
  auto MCBS = sycl::info::device::max_constant_buffer_size;
  (void)MCBS;
  // expected-error@+1{{no member named 'max_constant_args' in 'sycl::info::device'}}
  auto MCA = sycl::info::device::max_constant_args;
  (void)MCA;

  return 0;
}