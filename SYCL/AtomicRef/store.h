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
class store_kernel;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          typename T>
void store_test(queue q, size_t N) {
  T initial = T(N);
  T store = initial;
  {
    buffer<T> store_buf(&store, 1);
    q.submit([&](handler &cgh) {
      auto st = store_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<store_kernel<AtomicRef, T>>(
          range<1>(N), [=](item<1> it) {
            size_t gid = it.get_id(0);
            auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                                 access::address_space::global_space>(st[0]);
            atm.store(T(gid));
          });
    });
  }

  // The initial value should have been overwritten by a work-item ID
  // Atomicity isn't tested here, but support for store() is
  assert(store != initial);
  assert(store >= T(0) && store <= T(N - 1));
}

template <typename T> void store_test(queue q, size_t N) {
  store_test<::sycl::ext::oneapi::atomic_ref, T>(q, N);
  store_test<::sycl::atomic_ref, T>(q, N);
}
