// REQUIRES: linux, gpu && level_zero
// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// RUN: %{build} %device_asan_flags -O0 -g -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// Force the DMA_BUF handle type for this test
constexpr auto ExportHandleType = syclexp::external_mem_handle_type::dma_buf;

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12;
  int *array = (int *)syclexp::alloc_exportable_device_mem(0, N * sizeof(int),
                                                           ExportHandleType, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernelR_4>(
        sycl::nd_range<1>(N + 1, 1),
        [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
  });
  Q.wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Exportable Memory
  // CHECK: {{READ of size 4 at kernel <.*MyKernelR_4> LID\(0, 0, 0\) GID\(12, 0, 0\)}}
  // CHECK: {{  #0 .* .*check_export_mem.cpp:}}[[@LINE-5]]

  syclexp::free_exportable_memory(array, Q);

  return 0;
}