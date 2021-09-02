#pragma once

#include "atomic_memory_order.h"
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <typename T> class atomic_memory_order_acq_rel_kernel;

template <typename T> void acq_rel_test(queue q, size_t N) {
  T a = 0;
  {
    buffer<T> a_buf(&a, 1);

    q.submit([&](handler &cgh) {
      auto a_acc = a_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<atomic_memory_order_acq_rel_kernel<T>>(
          range<1>(N), [=](item<1> it) {
            auto aar =
                atomic_ref<T, memory_order::acq_rel, memory_scope::device,
                           access::address_space::global_space>(a_acc[0]);
            auto ld = aar.load();
            ld += 1;
            aar.store(ld);
          });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(a == T(N));
}
