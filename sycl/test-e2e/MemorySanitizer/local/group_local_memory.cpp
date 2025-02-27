// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -mllvm -msan-spir-privates=0 -g -O0 -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s --check-prefixes CHECK-O0
// RUN: %{build} %device_msan_flags -g -O2 -o %t2.out
// RUN: %{run} %t2.out 2>&1 | FileCheck %s

// XFAIL: spirv-backend && gpu && run-mode
// XFAIL-TRACKER: https://github.com/llvm/llvm-project/issues/122075

#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/queue.hpp>

constexpr std::size_t global_size = 16;
constexpr std::size_t local_size = 8;

///
/// sycl::group_local_memory provides SLM initializer, so we can't detect UUM
/// here. But when we build the program use "-O0", the pointer of local memory
/// will be saved into private memory, then the initialization of local memory
/// will be skip (since the address space of memset is 0).
///

__attribute__((noinline)) void check(int data) { (void)data; }

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          auto ptr = sycl::ext::oneapi::group_local_memory<int[global_size]>(
              item.get_group());
          auto &ref = *ptr;
          check(ref[item.get_local_linear_id()]);
        });
  });
  Q.wait();
  // CHECK-O0-NOT: [kernel]
  // CHECK-O0: DeviceSanitizer: use-of-uninitialized-value
  // CHECK-O0: #0 {{.*}} {{.*group_local_memory.cpp}}:[[@LINE-6]]

  std::cout << "PASS" << std::endl;
  return 0;
}
// CHECK-NOT: DeviceSanitizer: use-of-uninitialized-value
// CHECK: PASS
