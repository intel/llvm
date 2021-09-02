#pragma once

#include "atomic_memory_order.h"
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <typename T> class atomic_memory_order_seq_cst_kernel;

template <typename T> void seq_cst_test(queue q, size_t N) {
  T a = 0;
  T b = 0;
  {
    buffer<T> a_buf(&a, 1);
    buffer<T> b_buf(&b, 1);

    q.submit([&](handler &cgh) {
      auto a_acc = a_buf.template get_access<access::mode::read_write>(cgh);
      auto b_acc = b_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<atomic_memory_order_seq_cst_kernel<T>>(
          range<1>(N), [=](item<1> it) {
            auto aar =
                atomic_ref<T, memory_order::seq_cst, memory_scope::device,
                           access::address_space::global_space>(a_acc[0]);
            auto bar =
                atomic_ref<T, memory_order::seq_cst, memory_scope::device,
                           access::address_space::global_space>(b_acc[0]);
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
