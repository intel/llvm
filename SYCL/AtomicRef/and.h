#pragma once

#ifndef TEST_GENERIC_IN_LOCAL
#define TEST_GENERIC_IN_LOCAL 0
#endif

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void and_local_test(queue q) {
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
       accessor<T, 1, access::mode::read_write, access::target::local> loc(1,
                                                                           cgh);

       cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
         int gid = it.get_global_id(0);
         if (gid == 0)
           loc[0] = T((1ll << N) - 1);
         it.barrier(access::fence_space::local_space);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (loc[0]);
         out[gid] = atm.fetch_and(~T(1ll << gid), order);
         it.barrier(access::fence_space::local_space);
         if (gid == 0)
           cum[0] = loc[0];
       });
     }).wait_and_throw();
  }

  // Final value should be equal to 0
  assert(cum == 0);

  // All other values should be unique; each work-item sets one bit to 0
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void and_global_test(queue q) {
  const size_t N = 32;
  const T initial = T((1ll << N) - 1);
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
        out[gid] = atm.fetch_and(~T(1ll << gid), order);
      });
    });
  }

  // Final value should be equal to 0
  assert(cum == 0);

  // All other values should be unique; each work-item sets one bit to 0
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void and_test(queue q) {
  constexpr bool do_local_tests =
      space == access::address_space::local_space ||
      (space == access::address_space::generic_space && TEST_GENERIC_IN_LOCAL);
  constexpr bool do_global_tests =
      space == access::address_space::global_space ||
      (space == access::address_space::generic_space && !TEST_GENERIC_IN_LOCAL);
  constexpr bool do_ext_tests = space != access::address_space::generic_space;
  if constexpr (do_local_tests) {
    if constexpr (do_ext_tests) {
      and_local_test<::sycl::ext::oneapi::atomic_ref, space, T, order, scope>(
          q);
    }
    and_local_test<::sycl::atomic_ref, space, T, order, scope>(q);
  }
  if constexpr (do_global_tests) {
    if constexpr (do_ext_tests) {
      and_global_test<::sycl::ext::oneapi::atomic_ref, space, T, order, scope>(
          q);
    }
    and_global_test<::sycl::atomic_ref, space, T, order, scope>(q);
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed>
void and_test_scopes(queue q) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
#if defined(SYSTEM)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
      scopes.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  and_test<space, T, order, memory_scope::system>(q);
#elif defined(WORK_GROUP)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
      scopes.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  and_test<space, T, order, memory_scope::work_group>(q);
#elif defined(SUB_GROUP)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
      scopes.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  and_test<space, T, order, memory_scope::sub_group>(q);
#else
  and_test<space, T, order, memory_scope::device>(q);
#endif
}

template <access::address_space space, typename T>
void and_test_orders_scopes(queue q) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
#if defined(ACQ_REL)
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) ==
      orders.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  and_test_scopes<space, T, memory_order::acq_rel>(q);
#elif defined(ACQUIRE)
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) ==
      orders.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  and_test_scopes<space, T, memory_order::acquire>(q);
#elif defined(RELEASE)
  if (std::find(orders.begin(), orders.end(), memory_order::release) ==
      orders.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  and_test_scopes<space, T, memory_order::release>(q);
#else
  and_test_scopes<space, T, memory_order::relaxed>(q);
#endif
}

template <access::address_space space> void and_test_all() {
  queue q;

#ifdef ATOMIC64
  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return;
  }

  if constexpr (sizeof(long) == 8) {
    and_test_orders_scopes<space, long>(q);
    and_test_orders_scopes<space, unsigned long>(q);
  }
  if constexpr (sizeof(long long) == 8) {
    and_test_orders_scopes<space, long long>(q);
    and_test_orders_scopes<space, unsigned long long>(q);
  }
#else
  and_test_orders_scopes<space, int>(q);
  and_test_orders_scopes<space, unsigned int>(q);
  if constexpr (sizeof(long) == 4) {
    and_test_orders_scopes<space, long>(q);
    and_test_orders_scopes<space, unsigned long>(q);
  }
#endif

  std::cout << "Test passed." << std::endl;
}
