// UNSUPPORTED: true
// TODO: support dynamic local

// RUN: %{build} %device_sanitizer_flags -g -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %s
#include "sycl/accessor.hpp"
#include <cstddef>
#include <sycl/sycl.hpp>

constexpr std::size_t N = 1024;
constexpr std::size_t group_size = 16;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.parallel_for_work_group(N / group_size, group_size, [=]() {
      int array[N];
      group.parallel_for_work_item(
          [&](sycl::h_item<i> item) { ++array[item.get_global_id()]; });
    });
  });
  Q.wait();

  return 0;
}
