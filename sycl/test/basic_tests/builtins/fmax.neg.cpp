// RUN: %clangxx -fpreview-breaking-changes -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
#include <sycl/sycl.hpp>

decltype(auto) SYCL_EXTERNAL test_scalar(float x, double y) {
  // expected-error@+1 {{call to 'fmax' is ambiguous}}
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_vec(sycl::vec<float, 2> x,
                                      sycl::vec<double, 2> y) {
  // expected-error@+1 {{no matching function for call to 'fmax'}}
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_vec_marray(sycl::vec<float, 2> x,
                                             sycl::marray<double, 2> y) {
  // expected-error@+1 {{no matching function for call to 'fmax'}}
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_vec_marray(float x, sycl::vec<float, 2> y) {
  // expected-error@+1 {{no matching function for call to 'fmax'}}
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_marray_swizzle(sycl::marray<float, 2> x,
                                                 sycl::vec<float, 4> y) {
  // expected-error@+1 {{no matching function for call to 'fmax'}}
  return sycl::fmax(x, y.swizzle<2, 3>());
}
decltype(auto) SYCL_EXTERNAL test_swizzle_marray(sycl::vec<float, 4> x,
                                                 sycl::marray<float, 2> y) {
  // expected-error@+1 {{no matching function for call to 'fmax'}}
  return sycl::fmax(x.swizzle<2, 3>(), y);
}
