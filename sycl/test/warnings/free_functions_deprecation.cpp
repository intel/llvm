// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <CL/sycl.hpp>

int main() {
  // expected-warning@+1 {{'this_group<1>' is deprecated: use sycl::ext::oneapi::experimental::this_group() instead}}
  (void)sycl::this_group<1>();
  // expected-warning@+1 {{'this_item<1>' is deprecated: use sycl::ext::oneapi::experimental::this_item() instead}}
  (void)sycl::this_item<1>();
  // expected-warning@+1 {{'this_nd_item<1>' is deprecated: use sycl::ext::oneapi::experimental::this_nd_item() instead}}
  (void)sycl::this_nd_item<1>();
  // expected-warning@+1 {{'this_id<1>' is deprecated: use sycl::ext::oneapi::experimental::this_id() instead}}
  (void)sycl::this_id<1>();
  // expected-warning@+1 {{'this_sub_group' is deprecated: use sycl::ext::oneapi::experimental::this_sub_group() instead}}
  (void)sycl::ext::oneapi::this_sub_group();

  return 0;
}
