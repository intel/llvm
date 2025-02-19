// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -g -O0 -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O1 -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O2 -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 16;
constexpr std::size_t local_size = 8;

__attribute__((noinline)) void check(int data) { (void)data; }

__attribute__((noinline)) void foo(sycl::nd_item<1> &item) {
  auto ptr =
      sycl::ext::oneapi::group_local_memory<int[global_size]>(item.get_group());
  auto &ref = *ptr;
  check(ref[item.get_local_linear_id()]);
}

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(sycl::nd_range<1>(global_size, local_size),
                                   [=](sycl::nd_item<1> item) { foo(item); });
  });

  Q.wait();
  return 0;
}
