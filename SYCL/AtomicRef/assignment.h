#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <typename T> class assignment_kernel;

template <typename T> void assignment_test(queue q, size_t N) {
  T initial = T(N);
  T assignment = initial;
  {
    buffer<T> assignment_buf(&assignment, 1);
    q.submit([&](handler &cgh) {
      auto st =
          assignment_buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<assignment_kernel<T>>(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = atomic_ref<T, ONEAPI::memory_order::relaxed,
                              ONEAPI::memory_scope::device,
                              access::address_space::global_space>(st[0]);
        atm = T(gid);
      });
    });
  }

  // The initial value should have been overwritten by a work-item ID
  // Atomicity isn't tested here, but support for assignment() is
  assert(assignment != initial);
  assert(assignment >= T(0) && assignment <= T(N - 1));
}
