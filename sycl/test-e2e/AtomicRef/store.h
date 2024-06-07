#pragma once

#ifndef TEST_GENERIC_IN_LOCAL
#define TEST_GENERIC_IN_LOCAL 0
#endif

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include <sycl/detail/core.hpp>

#include <sycl/atomic_ref.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void store_global_test(queue q, size_t N) {
  T initial = T(N);
  T store = initial;
  {
    buffer<T> store_buf(&store, 1);
    q.submit([&](handler &cgh) {
      auto st = store_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, scope, space>(st[0]);
        atm.store(T(gid), order);
      });
    });
  }

  // The initial value should have been overwritten by a work-item ID
  // Atomicity isn't tested here, but support for store() is
  assert(store != initial);
  assert(store >= T(0) && store <= T(N - 1));
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void store_global_test_usm_shared(queue q, size_t N) {
  T initial = T(N);
  T *st = malloc_shared<T>(1, q);
  st[0] = initial;
  {
    q.submit([&](handler &cgh) {
       cgh.parallel_for(range<1>(N), [=](item<1> it) {
         size_t gid = it.get_id(0);
         auto atm = AtomicRef<T, memory_order::relaxed, scope, space>(st[0]);
         atm.store(T(gid), order);
       });
     }).wait_and_throw();
  }

  // The initial value should have been overwritten by a work-item ID.
  // Atomicity isn't tested here, but support for store() is.
  assert(st[0] != initial);
  assert(st[0] >= T(0) && st[0] <= T(N - 1));

  free(st, q);
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void store_local_test(queue q, size_t N) {
  T initial = T(N);
  T store = initial;
  {
    buffer<T> store_buf(&store, 1);
    q.submit([&](handler &cgh) {
      auto st = store_buf.template get_access<access::mode::read_write>(cgh);
      local_accessor<T, 1> loc(1, cgh);
      cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
        size_t gid = it.get_global_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, scope, space>(loc[0]);
        atm.store(T(gid), order);
        it.barrier(access::fence_space::local_space);
        if (gid == 0)
          st[0] = loc[0];
      });
    });
  }

  // The initial value should have been overwritten by a work-item ID
  // Atomicity isn't tested here, but support for store() is
  assert(store != initial);
  assert(store >= T(0) && store <= T(N - 1));
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void store_test(queue q, size_t N) {
  constexpr bool do_local_tests =
      space == access::address_space::local_space ||
      (space == access::address_space::generic_space && TEST_GENERIC_IN_LOCAL);
  constexpr bool do_global_tests =
      space == access::address_space::global_space ||
      (space == access::address_space::generic_space && !TEST_GENERIC_IN_LOCAL);
  constexpr bool do_ext_tests = space != access::address_space::generic_space;
  bool do_usm_tests = q.get_device().has(aspect::usm_shared_allocations);
  if constexpr (do_local_tests) {
#ifdef RUN_DEPRECATED
    if constexpr (do_ext_tests) {
      store_local_test<::sycl::ext::oneapi::atomic_ref, space, T, order, scope>(
          q, N);
    }
#else
    store_local_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
#endif
  }
  if constexpr (do_global_tests) {
#ifdef RUN_DEPRECATED
    if constexpr (do_ext_tests) {
      store_global_test<::sycl::ext::oneapi::atomic_ref, space, T, order,
                        scope>(q, N);
      if (do_usm_tests) {
        store_global_test_shared_usm<::sycl::ext::oneapi::atomic_ref, space, T,
                                     order, scope>(q, N);
      }
    }
#else
    store_global_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
    if (do_usm_tests) {
      store_global_test_usm_shared<::sycl::atomic_ref, space, T, order, scope>(
          q, N);
    }
#endif
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed>
void store_test_scopes(queue q, size_t N) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) !=
      scopes.end()) {
    store_test<space, T, order, memory_scope::system>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    store_test<space, T, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    store_test<space, T, order, memory_scope::sub_group>(q, N);
  }
  store_test<space, T, order, memory_scope::device>(q, N);
}

template <access::address_space space, typename T>
void store_test_orders_scopes(queue q, size_t N) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    store_test_scopes<space, T, memory_order::release>(q, N);
  }
  store_test_scopes<space, T, memory_order::relaxed>(q, N);
}

template <access::address_space space> void store_test_all() {
  queue q;

  constexpr int N = 32;
#ifdef FULL_ATOMIC64_COVERAGE
  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return;
  }

  store_test_orders_scopes<space, double>(q, N);
  if constexpr (sizeof(long) == 8) {
    store_test_orders_scopes<space, long>(q, N);
    store_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(long long) == 8) {
    store_test_orders_scopes<space, long long>(q, N);
    store_test_orders_scopes<space, unsigned long long>(q, N);
  }
  if constexpr (sizeof(char *) == 8) {
    store_test_orders_scopes<space, char *>(q, N);
  }
#endif
  store_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
  store_test_orders_scopes<space, int>(q, N);
  store_test_orders_scopes<space, unsigned int>(q, N);
  if constexpr (sizeof(long) == 4) {
    store_test_orders_scopes<space, long>(q, N);
    store_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(char *) == 4) {
    store_test_orders_scopes<space, char *>(q, N);
  }
#endif

  std::cout << "Test passed." << std::endl;
}
