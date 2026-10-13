// REQUIRES: linux, gpu && level_zero
// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// ALLOW_RETRIES: 10
// RUN: %{build} %device_tsan_flags -O0 -g -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s

// XFAIL: spirv-backend && arch-intel_gpu_pvc
// XFAIL-TRACKER: https://github.com/llvm/llvm-project/issues/160602

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr auto ExportHandleType = syclexp::external_mem_handle_type::dma_buf;

__attribute__((noinline)) void foo(char *array, int val) { *array += val; }

int main() {
  sycl::queue Q;
  auto *array =
      (char *)syclexp::alloc_exportable_device_mem(0, 1, ExportHandleType, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Test>(sycl::nd_range<1>(128, 8),
                                [=](sycl::nd_item<1> it) {
                                  *array += it.get_global_linear_id();
                                  foo(array, it.get_local_linear_id());
                                });
   }).wait();
  // CHECK: WARNING: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 1 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}check_export_mem.cpp

  syclexp::free_exportable_memory(array, Q);
  return 0;
}
