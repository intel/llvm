// REQUIRES: linux, gpu && level_zero
// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// Force the DMA_BUF handle type for this test
constexpr auto ExportHandleType = syclexp::external_mem_handle_type::dma_buf;

__attribute__((noinline)) int check(int data1, int data2) {
  return data1 + data2;
}

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 4;
  int *array = (int *)syclexp::alloc_exportable_device_mem(0, N * sizeof(int),
                                                           ExportHandleType, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>(
        [=]() { array[0] = check(array[0], array[1]); });
  });
  Q.wait();
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>
  // CHECK: #0 {{.*}} {{.*check_export_mem.cpp}}:[[@LINE-5]]

  syclexp::free_exportable_memory(array, Q);

  return 0;
}
