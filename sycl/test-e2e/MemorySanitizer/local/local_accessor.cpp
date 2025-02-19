// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -g -O0 -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O1 -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O2 -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s
#include <sycl/usm.hpp>

constexpr std::size_t global_size = 4;
constexpr std::size_t local_size = 1;

__attribute__((noinline)) int check(int data) { return data; }

int main() {
  sycl::queue Q;
  auto data = sycl::malloc_device<int>(global_size, Q);

  Q.submit([&](sycl::handler &cgh) {
    auto acc = sycl::local_accessor<int>(local_size, cgh);
    cgh.parallel_for<class MyKernel>(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          data[item.get_global_id()] = check(acc[item.get_local_id()]);
        });
  });

  Q.wait();
  sycl::free(data, Q);
  return 0;
}
