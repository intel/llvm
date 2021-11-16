#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          typename T>
class assignment_kernel;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          typename T>
void assignment_test(queue q, size_t N) {
  T initial = T(N);
  T assignment = initial;
  {
    buffer<T> assignment_buf(&assignment, 1);
    q.submit([&](handler &cgh) {
      auto st =
          assignment_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<assignment_kernel<AtomicRef, T>>(
          range<1>(N), [=](item<1> it) {
            size_t gid = it.get_id(0);
            auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                                 access::address_space::global_space>(st[0]);
            atm = T(gid);
          });
    });
  }

  // The initial value should have been overwritten by a work-item ID
  // Atomicity isn't tested here, but support for assignment() is
  assert(assignment != initial);
  assert(assignment >= T(0) && assignment <= T(N - 1));
}

template <typename T> void assignment_test(queue q, size_t N) {
  assignment_test<::sycl::ext::oneapi::atomic_ref, T>(q, N);
  assignment_test<::sycl::atomic_ref, T>(q, N);
}
