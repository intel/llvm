// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <CL/sycl.hpp>

int main() {
  cl::sycl::id<1> id_obj(64);
  // expected-warning@+1 {{'operator range' is deprecated: range() conversion is deprecated}}
  (void)(cl::sycl::range<1>) id_obj;

  return 0;
}