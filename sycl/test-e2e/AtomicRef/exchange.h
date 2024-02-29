#pragma once

#ifndef TEST_GENERIC_IN_LOCAL
#define TEST_GENERIC_IN_LOCAL 0
#endif

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void exchange_local_test(queue q, size_t N) {
  const T initial = T(N);
  T cum = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(123456));
  {
    buffer<T> cum_buf(&cum, 1);
    buffer<T> output_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
       auto cum = cum_buf.template get_access<access::mode::read_write>(cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       local_accessor<T, 1> loc(1, cgh);

       cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
         int gid = it.get_global_id(0);
         if (gid == 0)
           loc[0] = initial;
         it.barrier(access::fence_space::local_space);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (loc[0]);
         out[gid] = atm.exchange(T(gid), order);
         it.barrier(access::fence_space::local_space);
         if (gid == 0)
           cum[0] = loc[0];
       });
     }).wait_and_throw();
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be unique; each work-item replaces the value it
  // reads with its own ID
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void exchange_global_test(queue q, size_t N) {
  const T initial = T(N);
  T exchange = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> exchange_buf(&exchange, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto exc =
          exchange_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (exc[0]);
        out[gid] = atm.exchange(T(gid), order);
      });
    });
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be unique; each work-item replaces the value it
  // reads with its own ID
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void exchange_global_test_usm_shared(queue q, size_t N) {
  const T initial = T(N);
  T *exc = malloc_shared<T>(1, q);
  exc[0] = initial;
  T *output = malloc_shared<T>(N, q);
  T *output_begin = &output[0], *output_end = &output[N];
  std::fill(output_begin, output_end, T(0));
  {
    q.submit([&](handler &cgh) {
       cgh.parallel_for(range<1>(N), [=](item<1> it) {
         size_t gid = it.get_id(0);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (exc[0]);
         output[gid] = atm.exchange(T(gid), order);
       });
     }).wait_and_throw();
  }

  // Only one work-item should have received the initial sentinel value.
  assert(std::count(output_begin, output_end, initial) == 1);

  // All other values should be unique; each work-item replaces the value it
  // reads with its own ID
  std::sort(output_begin, output_end);
  assert(std::unique(output_begin, output_end) == output_end);
  for (int i = 0; i < N; ++i) {
    assert((output[i] >= 0 && output[i] < N) || output[i] == initial);
  }

  free(exc, q);
  free(output, q);
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void exchange_test(queue q, size_t N) {
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
      exchange_local_test<::sycl::ext::oneapi::atomic_ref, space, T, order,
                          scope>(q, N);
    }
#else
    exchange_local_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
#endif
  }
  if constexpr (do_global_tests) {
#ifdef RUN_DEPRECATED
    if constexpr (do_ext_tests) {
      exchange_global_test<::sycl::ext::oneapi::atomic_ref, space, T, order,
                           scope>(q, N);
      if (do_usm_tests) {
        exchange_global_test_usm_shared<::sycl::ext::oneapi::atomic_ref, space,
                                        T, order, scope>(q, N);
      }
    }
#else
    exchange_global_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
    if (do_usm_tests) {
      exchange_global_test_usm_shared<::sycl::atomic_ref, space, T, order,
                                      scope>(q, N);
    }
#endif
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed>
void exchange_test_scopes(queue q, size_t N) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    exchange_test<space, T, order, memory_scope::sub_group>(q, N);
  }
  exchange_test<space, T, order, memory_scope::device>(q, N);
}

template <access::address_space space, typename T>
void exchange_test_orders_scopes(queue q, size_t N) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::acq_rel>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::acquire>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    exchange_test_scopes<space, T, memory_order::release>(q, N);
  }
  exchange_test_scopes<space, T, memory_order::relaxed>(q, N);
}

template <access::address_space space> void exchange_test_all() {
  queue q;

  constexpr int N = 32;
#ifdef FULL_ATOMIC64_COVERAGE
  exchange_test_orders_scopes<space, double>(q, N);
  if constexpr (sizeof(long) == 8) {
    exchange_test_orders_scopes<space, long>(q, N);
    exchange_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(long long) == 8) {
    exchange_test_orders_scopes<space, long long>(q, N);
    exchange_test_orders_scopes<space, unsigned long long>(q, N);
  }
#endif
  exchange_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
  exchange_test_orders_scopes<space, int>(q, N);
  exchange_test_orders_scopes<space, unsigned int>(q, N);
  if constexpr (sizeof(long) == 4) {
    exchange_test_orders_scopes<space, long>(q, N);
    exchange_test_orders_scopes<space, unsigned long>(q, N);
  }
#endif

  std::cout << "Test passed." << std::endl;
}
