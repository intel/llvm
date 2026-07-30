// REQUIRES: linux, gpu && level_zero
// REQUIRES: aspect-ext_oneapi_exportable_device_mem

// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O0 -g -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-ORIGIN-STACK
// RUN: %{build} %device_msan_flags -Xarch_device -fsanitize-memory-track-origins=1 -O2 -g -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr auto ExportHandleType = syclexp::external_mem_handle_type::dma_buf;

__attribute__((noinline)) int check(int data) { return data; }

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 2;
  int *array = (int *)syclexp::alloc_exportable_device_mem(0, N * sizeof(int),
                                                           ExportHandleType, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { array[0] = check(array[1]); });
  });
  Q.wait();
  // CHECK-NOT: [kernel]
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>
  // CHECK: #{{.*}} {{.*check_export_mem.cpp}}:[[@LINE-6]]
  // CHECK: ORIGIN: Exportable Memory allocation
  // CHECK-ORIGIN-STACK: #{{.*}} {{.*check_export_mem.cpp}}:[[@LINE-12]]

  syclexp::free_exportable_memory(array, Q);
  return 0;
}
