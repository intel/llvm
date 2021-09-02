#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <typename T> class load_kernel;

template <typename T> void load_test(queue q, size_t N) {
  T initial = T(42);
  T load = initial;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> load_buf(&load, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto ld = load_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<load_kernel<T>>(range<1>(N), [=](item<1> it) {
        size_t gid = it.get_id(0);
        auto atm = atomic_ref<T, memory_order::relaxed, memory_scope::device,
                              access::address_space::global_space>(ld[0]);
        out[gid] = atm.load();
      });
    });
  }

  // All work-items should read the same value
  // Atomicity isn't tested here, but support for load() is
  assert(std::all_of(output.begin(), output.end(),
                     [&](T x) { return (x == initial); }));
}
