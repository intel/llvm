#pragma once

#ifndef TEST_GENERIC_IN_LOCAL
#define TEST_GENERIC_IN_LOCAL 0
#endif

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void sub_fetch_local_test(queue q, size_t N) {
  T sum = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(123456));
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
       auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
       auto out =
           output_buf.template get_access<access::mode::discard_write>(cgh);
       local_accessor<T, 1> loc(1, cgh);

       cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
         int gid = it.get_global_id(0);
         if (gid == 0)
           loc[0] = T(N);
         it.barrier(access::fence_space::local_space);
         auto atm = AtomicRef < T,
              (order == memory_order::acquire || order == memory_order::release)
                  ? memory_order::relaxed
                  : order,
              scope, space > (loc[0]);
         out[gid] = atm.fetch_sub(Difference(1), order);
         it.barrier(access::fence_space::local_space);
         if (gid == 0)
           sum[0] = loc[0];
       });
     }).wait_and_throw();
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(sum == T(0));

  // Fetch returns original value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void sub_fetch_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (val[0]);
        out[gid] = atm.fetch_sub(Difference(1), order);
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // Fetch returns original value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void sub_minus_equal_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (val[0]);
        out[gid] = atm -= Difference(1);
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // -= returns updated value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void sub_pre_dec_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (val[0]);
        out[gid] = --atm;
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // Pre-decrement returns updated value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void sub_post_dec_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef < T,
             (order == memory_order::acquire || order == memory_order::release)
                 ? memory_order::relaxed
                 : order,
             scope, space > (val[0]);
        out[gid] = atm--;
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // Post-decrement returns original value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device>
void sub_test(queue q, size_t N) {
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
      sub_fetch_local_test<::sycl::ext::oneapi::atomic_ref, space, T,
                           Difference, order, scope>(q, N);
    }
#else
    sub_fetch_local_test<::sycl::atomic_ref, space, T, Difference, order,
                         scope>(q, N);
#endif
  }
  if constexpr (do_global_tests) {
#ifdef RUN_DEPRECATED
    if constexpr (do_ext_tests) {
      sub_fetch_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                     order, scope>(q, N);
      sub_minus_equal_test<::sycl::ext::oneapi::atomic_ref, space, T,
                           Difference, order, scope>(q, N);
      if constexpr (!std::is_floating_point_v<T>) {
        sub_pre_dec_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                         order, scope>(q, N);
        sub_post_dec_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                          order, scope>(q, N);
      }
    }
#else
    sub_fetch_test<::sycl::atomic_ref, space, T, Difference, order, scope>(q,
                                                                           N);
    sub_minus_equal_test<::sycl::atomic_ref, space, T, Difference, order,
                         scope>(q, N);
    if constexpr (!std::is_floating_point_v<T>) {
      sub_pre_dec_test<::sycl::atomic_ref, space, T, Difference, order, scope>(
          q, N);
      sub_post_dec_test<::sycl::atomic_ref, space, T, Difference, order, scope>(
          q, N);
    }
#endif
  }
}

template <access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed>
void sub_test_scopes(queue q, size_t N) {
  std::vector<memory_scope> scopes =
      q.get_device().get_info<info::device::atomic_memory_scope_capabilities>();
  if (std::find(scopes.begin(), scopes.end(), memory_scope::work_group) !=
      scopes.end()) {
    sub_test<space, T, Difference, order, memory_scope::work_group>(q, N);
  }
  if (std::find(scopes.begin(), scopes.end(), memory_scope::sub_group) !=
      scopes.end()) {
    sub_test<space, T, Difference, order, memory_scope::sub_group>(q, N);
  }
  sub_test<space, T, Difference, order, memory_scope::device>(q, N);
}

template <access::address_space space, typename T, typename Difference = T>
void sub_test_orders_scopes(queue q, size_t N) {
  std::vector<memory_order> orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();
  if (std::find(orders.begin(), orders.end(), memory_order::acq_rel) !=
      orders.end()) {
    sub_test_scopes<space, T, Difference, memory_order::acq_rel>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::acquire) !=
      orders.end()) {
    sub_test_scopes<space, T, Difference, memory_order::acquire>(q, N);
  }
  if (std::find(orders.begin(), orders.end(), memory_order::release) !=
      orders.end()) {
    sub_test_scopes<space, T, Difference, memory_order::release>(q, N);
  }
  sub_test_scopes<space, T, Difference, memory_order::relaxed>(q, N);
}

template <access::address_space space> void sub_test_all() {
  queue q;

  constexpr int N = 32;
#ifdef FULL_ATOMIC64_COVERAGE
  sub_test_orders_scopes<space, double>(q, N);
  if constexpr (sizeof(long) == 8) {
    sub_test_orders_scopes<space, long>(q, N);
    sub_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(long long) == 8) {
    sub_test_orders_scopes<space, long long>(q, N);
    sub_test_orders_scopes<space, unsigned long long>(q, N);
  }
  if constexpr (sizeof(char *) == 8) {
    sub_test_orders_scopes<space, char *, ptrdiff_t>(q, N);
  }
#endif
  sub_test_orders_scopes<space, float>(q, N);
#ifdef FULL_ATOMIC32_COVERAGE
  sub_test_orders_scopes<space, int>(q, N);
  sub_test_orders_scopes<space, unsigned int>(q, N);
  if constexpr (sizeof(long) == 4) {
    sub_test_orders_scopes<space, long>(q, N);
    sub_test_orders_scopes<space, unsigned long>(q, N);
  }
  if constexpr (sizeof(char *) == 4) {
    sub_test_orders_scopes<space, char *, ptrdiff_t>(q, N);
  }
#endif

  std::cout << "Test passed." << std::endl;
}
