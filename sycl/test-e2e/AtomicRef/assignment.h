#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include <sycl/detail/core.hpp>

#include <sycl/atomic_ref.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
class assignment_kernel;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
void assignment_test(queue q, size_t N) {
  T initial = T(N);
  T assignment = initial;
  {
    buffer<T> assignment_buf(&assignment, 1);
    q.submit([&](handler &cgh) {
      auto st =
          assignment_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<assignment_kernel<AtomicRef, address_space, T>>(
          range<1>(N), [=](item<1> it) {
            size_t gid = it.get_id(0);
            auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                                 address_space>(st[0]);
            atm = T(gid);
          });
    });
  }

  // The initial value should have been overwritten by a work-item ID
  // Atomicity isn't tested here, but support for assignment() is
  assert(assignment != initial);
  assert(assignment >= T(0) && assignment <= T(N - 1));
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
class assignment_usm_kernel;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
void assignment_test_usm_shared(queue q, size_t N) {
  T initial = T(N);
  T *st = malloc_shared<T>(1, q);
  st[0] = initial;
  {
    q.submit([&](handler &cgh) {
       cgh.parallel_for<assignment_usm_kernel<AtomicRef, address_space, T>>(
           range<1>(N), [=](item<1> it) {
             size_t gid = it.get_id(0);
             auto atm = AtomicRef<T, memory_order::relaxed,
                                  memory_scope::device, address_space>(st[0]);
             atm = T(gid);
           });
     }).wait_and_throw();
  }

  // The initial value should have been overwritten by a work-item ID.
  // Atomicity isn't tested here, but support for assignment() is.
  assert(st[0] != initial);
  assert(st[0] >= T(0) && st[0] <= T(N - 1));

  free(st, q);
}

template <typename T> void assignment_test(queue q, size_t N) {
  bool do_usm_tests = q.get_device().has(aspect::usm_shared_allocations);
#ifdef RUN_DEPRECATED
  assignment_test<::sycl::ext::oneapi::atomic_ref,
                  access::address_space::global_space, T>(q, N);
  if (do_usm_tests) {
    assignment_test_usm_shared<::sycl::ext::oneapi::atomic_ref,
                               access::address_space::global_space, T>(q, N);
  }
#else
  assignment_test<::sycl::atomic_ref, access::address_space::global_space, T>(
      q, N);
  if (do_usm_tests) {
    assignment_test_usm_shared<::sycl::atomic_ref,
                               access::address_space::global_space, T>(q, N);
  }
#endif
}

template <typename T> void assignment_generic_test(queue q, size_t N) {
  bool do_usm_tests = q.get_device().has(aspect::usm_shared_allocations);
  assignment_test<::sycl::atomic_ref, access::address_space::generic_space, T>(
      q, N);
  if (do_usm_tests) {
    assignment_test_usm_shared<::sycl::atomic_ref,
                               access::address_space::generic_space, T>(q, N);
  }
}
