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
class exchange_kernel;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T>
void exchange_test(queue q, size_t N) {
  const T initial = T(N);
  T exchange = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> exchange_buf(&exchange, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto exc =
          exchange_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<exchange_kernel<AtomicRef, address_space, T>>(
          range<1>(N), [=](item<1> it) {
            size_t gid = it.get_id(0);
            auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                                 address_space>(exc[0]);
            out[gid] = atm.exchange(T(gid));
          });
    });
  }

  // Only one work-item should have received the initial sentinel value
  assert(std::count(output.begin(), output.end(), initial) == 1);

  // All other values should be unique; each work-item replaces the value it
  // reads with its own ID
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T> void exchange_test(queue q, size_t N) {
  exchange_test<::sycl::ext::oneapi::atomic_ref,
                access::address_space::global_space, T>(q, N);
  exchange_test<::sycl::atomic_ref, access::address_space::global_space, T>(q,
                                                                            N);
}

template <typename T> void exchange_generic_test(queue q, size_t N) {
  exchange_test<::sycl::atomic_ref, access::address_space::generic_space, T>(q,
                                                                             N);
}
