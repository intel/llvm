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
void load_local_test(queue q, size_t N) {
  T initial = T(42);
  T load = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> load_buf(&load, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
       auto ld = load_buf.template get_access<access::mode::read_write>(cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       accessor<T, 1, access::mode::read_write, access::target::local> loc(1,
                                                                           cgh);
       cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
         int gid = it.get_global_id(0);
         if (gid == 0)
           loc[0] = initial;
         it.barrier(access::fence_space::local_space);
         auto atm = AtomicRef<T, memory_order::relaxed, scope, space>(loc[0]);
         out[gid] = atm.load(order);
       });
     }).wait_and_throw();
  }

  // All work-items should read the same value
  // Atomicity isn't tested here, but support for load() is
  assert(std::all_of(output.begin(), output.end(),
                     [&](T x) { return (x == initial); }));
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void load_global_test(queue q, size_t N) {
  T initial = T(42);
  T load = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> load_buf(&load, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto ld = load_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, scope, space>(ld[0]);
        out[gid] = atm.load(order);
      });
    });
  }

  // All work-items should read the same value
  // Atomicity isn't tested here, but support for load() is
  assert(std::all_of(output.begin(), output.end(),
                     [&](T x) { return (x == initial); }));
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void load_test(queue q, size_t N) {
  constexpr bool do_local_tests =
      space == access::address_space::local_space ||
      (space == access::address_space::generic_space && TEST_GENERIC_IN_LOCAL);
  constexpr bool do_global_tests =
      space == access::address_space::global_space ||
      (space == access::address_space::generic_space && !TEST_GENERIC_IN_LOCAL);
  constexpr bool do_ext_tests = space != access::address_space::generic_space;
  if constexpr (do_local_tests) {
#ifdef RUN_DEPRECATED
    if constexpr (do_ext_tests) {
      load_local_test<::sycl::ext::oneapi::atomic_ref, space, T, order, scope>(
          q, N);
    }
#else
    load_local_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
#endif
  }
  if constexpr (do_global_tests) {
#ifdef RUN_DEPRECATED
    if constexpr (do_ext_tests) {
      load_global_test<::sycl::ext::oneapi::atomic_ref, space, T, order, scope>(
          q, N);
    }
#else
    load_global_test<::sycl::atomic_ref, space, T, order, scope>(q, N);
#endif
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed>
void load_test_scopes(queue q, size_t N) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    load_test<space, T, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    load_test<space, T, order, memory_scope::sub_group>(q, N);
  }
  load_test<space, T, order, memory_scope::device>(q, N);
}

template <access::address_space space, typename T>
void load_test_orders_scopes(queue q, size_t N) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    load_test_scopes<space, T, memory_order::acquire>(q, N);
  }
  load_test_scopes<space, T, memory_order::relaxed>(q, N);
}

template <access::address_space space> void load_test_all() {
  queue q;

  constexpr int N = 32;
#ifdef FULL_ATOMIC64_COVERAGE
  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping atomic64 tests\n";
    return;
  }

  load_test_orders_scopes<space, double>(q, N);
  if constexpr (sizeof(long) == 8) {
    load_test_orders_scopes<space, long>(q, N);
    load_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(long long) == 8) {
    load_test_orders_scopes<space, long long>(q, N);
    load_test_orders_scopes<space, unsigned long long>(q, N);
  }
  if constexpr (sizeof(char *) == 8) {
    load_test_orders_scopes<space, char *>(q, N);
  }
#endif
  load_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
  load_test_orders_scopes<space, int>(q, N);
  load_test_orders_scopes<space, unsigned int>(q, N);
  if constexpr (sizeof(long) == 4) {
    load_test_orders_scopes<space, long>(q, N);
    load_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(char *) == 4) {
    load_test_orders_scopes<space, char *>(q, N);
  }
#endif

  std::cout << "Test passed." << std::endl;
}
