#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
void min_test(queue q, size_t N) {
  T initial = std::numeric_limits<T>::max();
  T val = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                             address_space>(val[0]);
        out[gid] = atm.fetch_min(T(gid));
      });
    });
  }

  // Final value should be equal to 0
  assert(val == 0);

  // Only one work-item should have received the initial value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // fetch_min returns original value
  // Intermediate values should all be <= initial value
  for (int i = 0; i < N; ++i) {
    assert(output[i] <= initial);
  }
}

template <typename T> void min_test(queue q, size_t N) {
  min_test<::sycl::ext::oneapi::atomic_ref, access::address_space::global_space,
           T>(q, N);
  min_test<::sycl::atomic_ref, access::address_space::global_space, T>(q, N);
}

template <typename T> void min_generic_test(queue q, size_t N) {
  min_test<::sycl::atomic_ref, access::address_space::generic_space, T>(q, N);
}
