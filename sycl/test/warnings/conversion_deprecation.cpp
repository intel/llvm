// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

int main() {
  sycl::id<1> id_obj(64);
  // expected-warning@+1 {{'operator range' is deprecated: range() conversion is deprecated}}
  (void)(sycl::range<1>)id_obj;

  return 0;
}
