#pragma once

#include "atomic_memory_order.h"
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
class atomic_memory_order_seq_cst_kernel;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
void seq_cst_test(queue q, size_t N) {
  T a = 0;
  T b = 0;
  {
    buffer<T> a_buf(&a, 1);
    buffer<T> b_buf(&b, 1);

    q.submit([&](handler &cgh) {
      auto a_acc = a_buf.template get_access<access::mode::read_write>(cgh);
      auto b_acc = b_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<
          atomic_memory_order_seq_cst_kernel<AtomicRef, address_space, T>>(
          range<1>(N), [=](item<1> it) {
            auto aar = AtomicRef<T, memory_order::seq_cst, memory_scope::device,
                                 address_space>(a_acc[0]);
            auto bar = AtomicRef<T, memory_order::seq_cst, memory_scope::device,
                                 address_space>(b_acc[0]);
            auto ald = aar.load();
            auto bld = bar.load();
            ald += 1;
            bld += ald;
            bar.store(bld);
            aar.store(ald);
          });
    });
  }

  // All work-items increment a by 1, so final value should be equal to N
  assert(a == T(N));
  // b is the sum of [1..N]
  size_t rsum = 0;
  for (size_t i = 1; i <= N; ++i)
    rsum += i;
  assert(b == T(rsum));
}

template <typename T> void seq_cst_test(queue q, size_t N) {
  seq_cst_test<::sycl::ext::oneapi::atomic_ref,
               access::address_space::global_space, T>(q, N);
  seq_cst_test<::sycl::atomic_ref, access::address_space::global_space, T>(q,
                                                                           N);
}

template <typename T> void seq_cst_generic_test(queue q, size_t N) {
  seq_cst_test<::sycl::atomic_ref, access::address_space::generic_space, T>(q,
                                                                            N);
}
