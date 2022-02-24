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
void compare_exchange_local_test(queue q, size_t N) {
  const T initial = T(N);
  T compare_exchange = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(123456));
  {
    buffer<T> compare_exchange_buf(&compare_exchange, 1);
    buffer<T> output_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
       auto compare_exchange =
           compare_exchange_buf.template get_access<access::mode::read_write>(
               cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       accessor<T, 1, access::mode::read_write, access::target::local> loc(1,
                                                                           cgh);

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
         T result = T(N); // Avoid copying pointer
         bool success = atm.compare_exchange_strong(result, (T)gid, order);
         if (success) {
           out[gid] = result;
         } else {
           out[gid] = T(gid);
         }
         it.barrier(access::fence_space::local_space);
         if (gid == 0)
           compare_exchange[0] = loc[0];
       });
     }).wait_and_throw();
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be the index itself or the sentinel value
  for (size_t i = 0; i < N; ++i) {
    assert(output[i] == T(i) || output[i] == initial);
  }
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void compare_exchange_global_test(queue q, size_t N) {
  const T initial = T(N);
  T compare_exchange = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> compare_exchange_buf(&compare_exchange, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
       auto exc =
           compare_exchange_buf.template get_access<access::mode::read_write>(
               cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       cgh.parallel_for(range<1>(N), [=](item<1> it) {
         size_t gid = it.get_id(0);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (exc[0]);
         T result = T(N); // Avoid copying pointer
         bool success = atm.compare_exchange_strong(result, (T)gid, order);
         if (success) {
           out[gid] = result;
         } else {
           out[gid] = T(gid);
         }
       });
     }).wait_and_throw();
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be the index itself or the sentinel value
  for (size_t i = 0; i < N; ++i) {
    assert(output[i] == T(i) || output[i] == initial);
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void compare_exchange_test(queue q, size_t N) {
  constexpr bool do_local_tests =
      space == access::address_space::local_space ||
      (space == access::address_space::generic_space && TEST_GENERIC_IN_LOCAL);
  constexpr bool do_global_tests =
      space == access::address_space::global_space ||
      (space == access::address_space::generic_space && !TEST_GENERIC_IN_LOCAL);
  constexpr bool do_ext_tests = space != access::address_space::generic_space;
  if constexpr (do_local_tests) {
    if constexpr (do_ext_tests) {
      compare_exchange_local_test<::sycl::ext::oneapi::atomic_ref, space, T,
                                  order, scope>(q, N);
    }
    compare_exchange_local_test<::sycl::atomic_ref, space, T, order, scope>(q,
                                                                            N);
  }
  if constexpr (do_global_tests) {
    if constexpr (do_ext_tests) {
      compare_exchange_global_test<::sycl::ext::oneapi::atomic_ref, space, T,
                                   order, scope>(q, N);
    }
    compare_exchange_global_test<::sycl::atomic_ref, space, T, order, scope>(q,
                                                                             N);
  }
}

template <access::address_space space, typename T,
          memory_order order = memory_order::relaxed>
void compare_exchange_test_scopes(queue q, size_t N) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
#if defined(SYSTEM)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
      scopes.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  compare_exchange_test<space, T, order, memory_scope::system>(q, N);
#elif defined(WORK_GROUP)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
      scopes.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  compare_exchange_test<space, T, order, memory_scope::work_group>(q, N);
#elif defined(SUB_GROUP)
  if (std::find(scopes.begin(), scopes.end(), memory_scope::system) ==
      scopes.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  compare_exchange_test<space, T, order, memory_scope::sub_group>(q, N);
#else
  compare_exchange_test<space, T, order, memory_scope::device>(q, N);
#endif
}

template <access::address_space space, typename T>
void compare_exchange_test_orders_scopes(queue q, size_t N) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
#if defined(ACQ_REL)
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) ==
      orders.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  compare_exchange_test_scopes<space, T, memory_order::acq_rel>(q, N);
#elif defined(ACQUIRE)
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) ==
      orders.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  compare_exchange_test_scopes<space, T, memory_order::acquire>(q, N);
#elif defined(RELEASE)
  if (std::find(orders.begin(), orders.end(), memory_order::release) ==
      orders.end()) {
    std::cout << "Skipping test\n";
    return;
  }
  compare_exchange_test_scopes<space, T, memory_order::release>(q, N);
#else
  compare_exchange_test_scopes<space, T, memory_order::relaxed>(q, N);
#endif
}

template <access::address_space space> void compare_exchange_test_all() {
  queue q;

  constexpr int N = 32;
#ifdef ATOMIC64
  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return;
  }
  compare_exchange_test_orders_scopes<space, double>(q, N);
  if constexpr (sizeof(long) == 8) {
    compare_exchange_test_orders_scopes<space, long>(q, N);
    compare_exchange_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(long long) == 8) {
    compare_exchange_test_orders_scopes<space, long long>(q, N);
    compare_exchange_test_orders_scopes<space, unsigned long long>(q, N);
  }
  if constexpr (sizeof(char *) == 8) {
    compare_exchange_test_orders_scopes<space, char *>(q, N);
  }
#else
  compare_exchange_test_orders_scopes<space, int>(q, N);
  compare_exchange_test_orders_scopes<space, unsigned int>(q, N);
  compare_exchange_test_orders_scopes<space, float>(q, N);

  if constexpr (sizeof(long) == 4) {
    compare_exchange_test_orders_scopes<space, long>(q, N);
    compare_exchange_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(char *) == 4) {
    compare_exchange_test_orders_scopes<space, char *>(q, N);
  }
#endif

  std::cout << "Test passed." << std::endl;
}
