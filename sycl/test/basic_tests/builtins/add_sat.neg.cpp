// RUN: %clangxx -fpreview-breaking-changes -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
#include <sycl/sycl.hpp>

decltype(auto) SYCL_EXTERNAL test(int x, unsigned int y) {
  // expected-error@+1 {{no matching function for call to 'add_sat'}}
  return sycl::add_sat(x, y);
}

decltype(auto) SYCL_EXTERNAL test(int x, short y) {
  // expected-error@+1 {{no matching function for call to 'add_sat'}}
  return sycl::add_sat(x, y);
}

decltype(auto) SYCL_EXTERNAL test(sycl::vec<int, 2> x, sycl::vec<int, 4> y) {
  // expected-error@+1 {{no matching function for call to 'add_sat'}}
  return sycl::add_sat(x, y);
}

decltype(auto) SYCL_EXTERNAL test(sycl::vec<int, 2> x, sycl::vec<int, 4> y) {
  // expected-error@+1 {{no matching function for call to 'add_sat'}}
  return sycl::add_sat(x, y.swizzle<0, 2, 1>());
}

decltype(auto) SYCL_EXTERNAL test(sycl::vec<int, 2> x, sycl::marray<int, 2> y) {
  // expected-error@+1 {{no matching function for call to 'add_sat'}}
  return sycl::add_sat(x, y);
}

decltype(auto) SYCL_EXTERNAL test(sycl::vec<int, 4> x, sycl::vec<int, 4> y) {
  // expected-error@+1 {{no matching function for call to 'add_sat'}}
  return sycl::add_sat(x.swizzle<2, 0>(), y.swizzle<2, 0, 1>());
}
