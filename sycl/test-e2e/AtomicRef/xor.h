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
void xor_local_test(queue q) {
  const size_t N = 32;
  T cum = 0;
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
           loc[0] = 0;
         it.barrier(access::fence_space::local_space);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (loc[0]);
         out[gid] = atm.fetch_xor(T(1ll << gid), order);
         it.barrier(access::fence_space::local_space);
         if (gid == 0)
           cum[0] = loc[0];
       });
     }).wait_and_throw();
  }

  // Final value should be equal to N ones
  assert(cum == T((1ll << N) - 1));

  // All other values should be unique; each wxork-item sets one bit to 1
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void xor_global_test(queue q) {
  const size_t N = 32;
  const T initial = 0;
  T cum = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> cum_buf(&cum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto cum = cum_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (cum[0]);
        out[gid] = atm.fetch_xor(T(1ll << gid), order);
      });
    });
  }

  // Final value should be equal to N ones
  assert(cum == T((1ll << N) - 1));

  // All other values should be unique; each wxork-item sets one bit to 1
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void xor_test(queue q) {
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
      xor_local_test<::sycl::ext::oneapi::atomic_ref, space, T, order, scope>(
          q);
    }
#else
    xor_local_test<::sycl::atomic_ref, space, T, order, scope>(q);
#endif
  }
  if constexpr (do_global_tests) {
#ifdef RUN_DEPRECATED
    if constexpr (do_ext_tests) {
      xor_global_test<::sycl::ext::oneapi::atomic_ref, space, T, order, scope>(
          q);
    }
#else
    xor_global_test<::sycl::atomic_ref, space, T, order, scope>(q);
#endif
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed>
void xor_test_scopes(queue q) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    xor_test<space, T, order, memory_scope::work_group>(q);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    xor_test<space, T, order, memory_scope::sub_group>(q);
  }
  xor_test<space, T, order, memory_scope::device>(q);
}

template <access::address_space space, typename T>
void xor_test_orders_scopes(queue q) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) !=
      orders.end()) {
    xor_test_scopes<space, T, memory_order::acq_rel>(q);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    xor_test_scopes<space, T, memory_order::acquire>(q);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    xor_test_scopes<space, T, memory_order::release>(q);
  }
  xor_test_scopes<space, T, memory_order::relaxed>(q);
}

template <access::address_space space> void xor_test_all() {
  queue q;

  constexpr int N = 32;
#ifdef FULL_ATOMIC64_COVERAGE
  if constexpr (sizeof(long) == 8) {
    xor_test_orders_scopes<space, long>(q);
    xor_test_orders_scopes<space, unsigned long>(q);
  }
  if constexpr (sizeof(long long) == 8) {
    xor_test_orders_scopes<space, long long>(q);
    xor_test_orders_scopes<space, unsigned long long>(q);
  }
#endif
  xor_test_orders_scopes<space, int>(q);
#ifdef FULL_ATOMIC32_COVERAGE
  xor_test_orders_scopes<space, unsigned int>(q);
  if constexpr (sizeof(long) == 4) {
    xor_test_orders_scopes<space, long>(q);
    xor_test_orders_scopes<space, unsigned long>(q);
  }
#endif

  std::cout << "Test passed." << std::endl;
}
