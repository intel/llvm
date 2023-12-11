// RUN: %clangxx -fpreview-breaking-changes -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
#include <sycl/sycl.hpp>
decltype(auto) SYCL_EXTERNAL test_vec_double(sycl::vec<double, 2> x) {
  // expected-error@+1 {{no matching function for call to 'cos'}}
  return sycl::native::cos(x);
}

decltype(auto) SYCL_EXTERNAL test_vec_half(sycl::vec<sycl::half, 2> x) {
  // expected-error@+1 {{no matching function for call to 'cos'}}
  return sycl::native::cos(x);
}
