// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -g -O0 -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O1 -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -g -O2 -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

// XFAIL: spirv-backend && gpu && run-mode
// XFAIL-TRACKER: https://github.com/llvm/llvm-project/issues/122075

#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/usm.hpp>

constexpr std::size_t global_size = 4;
constexpr std::size_t local_size = 1;

__attribute__((noinline)) int check(int data, sycl::nd_item<1> &item) {
  auto ptr =
      sycl::ext::oneapi::group_local_memory<int[global_size]>(item.get_group());
  auto &ref = *ptr;
  return data + ref[0];
}

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &cgh) {
    auto acc = sycl::local_accessor<int>(local_size, cgh);
    cgh.parallel_for<class MyKernel>(
        sycl::nd_range<1>(global_size, local_size),
        [=](sycl::nd_item<1> item) { check(acc[item.get_local_id()], item); });
  });
  Q.wait();
  // CHECK-NOT: [kernel]
  // CHECK: DeviceSanitizer: use-of-uninitialized-value
  // CHECK: #0 {{.*}} {{.*local_accessor.cpp}}:[[@LINE-5]]

  return 0;
}
