#pragma once

#ifndef TEST_GENERIC_IN_LOCAL
#define TEST_GENERIC_IN_LOCAL 0
#endif

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#include <sycl/detail/core.hpp>

#include <sycl/atomic_ref.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/half_type.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

template <template <typename, memory_order, memory_scope,
                    access::address_space> class AtomicRef,
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
       auto sum = sum_buf.template get_access<access_mode::read_write>(cgh);
       auto out = output_buf.get_access(cgh, sycl::write_only, sycl::no_init);
       local_accessor<T, 1> loc(1, cgh);

       cgh.parallel_for(nd_range<1>(N, N), [=](nd_item<1> it) {
         int gid = it.get_global_id(0);
         if (gid == 0)
           loc[0] = T(N);
         group_barrier(it.get_group());
         auto atm = AtomicRef<T,
                              (order == memory_order::acquire ||
                               order == memory_order::release)
                                  ? memory_order::relaxed
                                  : order,
                              scope, space>(loc[0]);
         out[gid] = atm.fetch_sub(Difference(1), order);
         group_barrier(it.get_group());
         if (gid == 0)
           sum[0] = loc[0];
       });
     }).wait_and_throw();
  }

  assert(sum == T(0));

  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T>
void sub_check(const T &val, T *output, size_t N, bool returns_original) {
  assert(val == T(0));

  T *output_end = output + N;
  auto min_e = std::min_element(output, output_end);
  auto max_e = std::max_element(output, output_end);
  if (returns_original)
    assert(*min_e == T(1) && *max_e == T(N));
  else
    assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output, output_end);
  assert(std::unique(output, output_end) == output_end);
}

template <typename T>
inline constexpr bool sub_has_inc_dec =
    !std::is_floating_point_v<T> && !std::is_same_v<T, sycl::half> &&
    !std::is_same_v<T, sycl::ext::oneapi::bfloat16>;

// Number of operations tested on global memory for type T:
// fetch_sub, operator-=, and (for types having them) pre/post decrement.
template <typename T>
inline constexpr size_t sub_num_ops = sub_has_inc_dec<T> ? 4 : 2;


template <template <typename, memory_order, memory_scope,
                    access::address_space> class AtomicRef,
          access::address_space space, typename T, typename Difference,
          memory_order order, memory_scope scope, typename ValT, typename OutT>
void sub_ops_body(ValT &val, OutT &out, size_t gid, size_t N) {
  constexpr memory_order atm_order =
      (order == memory_order::acquire || order == memory_order::release)
          ? memory_order::relaxed
          : order;
  using Atomic = AtomicRef<T, atm_order, scope, space>;
  {
    Atomic atm(val[0]);
    out[gid] = atm.fetch_sub(Difference(1), order);
  }
  {
    Atomic atm(val[1]);
    out[N + gid] = atm -= Difference(1);
  }
  if constexpr (sub_has_inc_dec<T>) {
    {
      Atomic atm(val[2]);
      out[2 * N + gid] = --atm;
    }
    {
      Atomic atm(val[3]);
      out[3 * N + gid] = atm--;
    }
  }
}

template <typename T> void sub_check_all(const T *val, T *output, size_t N) {
  sub_check<T>(val[0], output, N, /*returns_original=*/true);
  sub_check<T>(val[1], output + N, N, /*returns_original=*/false);
  if constexpr (sub_has_inc_dec<T>) {
    sub_check<T>(val[2], output + 2 * N, N, /*returns_original=*/false);
    sub_check<T>(val[3], output + 3 * N, N, /*returns_original=*/true);
  }
}

template <template <typename, memory_order, memory_scope,
                    access::address_space> class AtomicRef,
          access::address_space space, typename T, typename Difference = T,
          memory_order order = memory_order::relaxed,
          memory_scope scope = memory_scope::device, bool WithUsm = true>
void sub_global_test(queue q, size_t N) {
  constexpr size_t num_ops = sub_num_ops<T>;
  std::vector<T> val(num_ops, T(N));
  std::vector<T> output(num_ops * N, T(0));
  T *usm_val = nullptr;
  T *usm_output = nullptr;
  if constexpr (WithUsm) {
    usm_val = malloc_shared<T>(num_ops, q);
    std::fill(usm_val, usm_val + num_ops, T(N));
    usm_output = malloc_shared<T>(num_ops * N, q);
    std::fill(usm_output, usm_output + num_ops * N, T(0));
  }
  {
    buffer<T> val_buf(val.data(), val.size());
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
       auto val = val_buf.template get_access<access_mode::read_write>(cgh);
       auto out = output_buf.get_access(cgh, sycl::write_only, sycl::no_init);
       cgh.parallel_for(range<1>(N), [=](item<1> it) {
         sub_ops_body<AtomicRef, space, T, Difference, order, scope>(
             val, out, it.get_id(0), N);
         if constexpr (WithUsm)
           sub_ops_body<AtomicRef, space, T, Difference, order, scope>(
               usm_val, usm_output, it.get_id(0), N);
       });
     }).wait_and_throw();
  }
  sub_check_all<T>(val.data(), output.data(), N);
  if constexpr (WithUsm) {
    sub_check_all<T>(usm_val, usm_output, N);
    free(usm_val, q);
    free(usm_output, q);
  }
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
  bool do_usm_tests = q.get_device().has(aspect::usm_shared_allocations);
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
      sub_global_test<::sycl::ext::oneapi::atomic_ref, space, T, Difference,
                      order, scope, /*WithUsm=*/true>(q, N);
    }
#else
    if (do_usm_tests)
      sub_global_test<::sycl::atomic_ref, space, T, Difference, order, scope,
                      /*WithUsm=*/true>(q, N);
    else
      sub_global_test<::sycl::atomic_ref, space, T, Difference, order, scope,
                      /*WithUsm=*/false>(q, N);
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

#ifdef FULL_ATOMIC16_COVERAGE
  if (!q.get_device().has(aspect::ext_oneapi_atomic16)) {
    std::cout << "Skipping ext_oneapi_atomic16 tests\n";
    return;
  }

  sub_test_orders_scopes<space, short>(q, N);
  sub_test_orders_scopes<space, unsigned short>(q, N);
  sub_test_orders_scopes<space, sycl::half>(q, N);
  sub_test_orders_scopes<space, sycl::ext::oneapi::bfloat16>(q, N);
#endif

  std::cout << "Test passed." << std::endl;
}
