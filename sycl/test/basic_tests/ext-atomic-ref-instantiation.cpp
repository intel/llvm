// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note

#include <sycl/ext/oneapi/atomic_ref.hpp>

struct A {};

int main() {
  double d = 10.0;
  auto ref_d =
      sycl::ext::oneapi::atomic_ref<double,
                                    sycl::ext::oneapi::memory_order_acq_rel,
                                    sycl::ext::oneapi::memory_scope_device,
                                    sycl::access::address_space::local_space>(
          d);
  // expected-warning@-5 {{is deprecated: use 'sycl::atomic_ref' instead}}
  // expected-warning@-5 {{'memory_order_acq_rel' is deprecated: use 'sycl::memory_order_acq_rel' instead}}
  // expected-warning@-5 {{'memory_scope_device' is deprecated: use 'sycl::memory_scope_device' instead}}

  int i = 10;
  auto ref_i =
      sycl::ext::oneapi::atomic_ref<int,
                                    sycl::ext::oneapi::memory_order_acq_rel,
                                    sycl::ext::oneapi::memory_scope_device,
                                    sycl::access::address_space::local_space>(
          i);
  // expected-warning@-5 {{is deprecated: use 'sycl::atomic_ref' instead}}
  // expected-warning@-5 {{'memory_order_acq_rel' is deprecated: use 'sycl::memory_order_acq_rel' instead}}
  // expected-warning@-5 {{'memory_scope_device' is deprecated: use 'sycl::memory_scope_device' instead}}

  A a;
  A *p = &a;
  auto ref_p =
      sycl::ext::oneapi::atomic_ref<A *,
                                    sycl::ext::oneapi::memory_order_acq_rel,
                                    sycl::ext::oneapi::memory_scope_device,
                                    sycl::access::address_space::local_space>(
          p);
  // expected-warning@-5 {{is deprecated: use 'sycl::atomic_ref' instead}}
  // expected-warning@-5 {{'memory_order_acq_rel' is deprecated: use 'sycl::memory_order_acq_rel' instead}}
  // expected-warning@-5 {{'memory_scope_device' is deprecated: use 'sycl::memory_scope_device' instead}}

  return 0;
}
