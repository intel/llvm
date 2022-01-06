// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -o %t.out -Xclang -verify-ignore-unexpected=note
// expected-no-diagnostics
#include <CL/sycl/atomic_ref.hpp>

struct A {};

int main() {
  double d = 10.0;
  auto ref_d = sycl::atomic_ref<double, sycl::memory_order_acq_rel,
                                sycl::memory_scope_device,
                                sycl::access::address_space::local_space>(d);


  int i = 10;
  auto ref_i = sycl::atomic_ref<int, sycl::memory_order_acq_rel,
                                sycl::memory_scope_device,
                                sycl::access::address_space::local_space>(i);

  A a;
  A* p = &a;
  auto ref_p = sycl::atomic_ref<A *, sycl::memory_order_acq_rel,
                                sycl::memory_scope_device>(p);
  return 0;
}
