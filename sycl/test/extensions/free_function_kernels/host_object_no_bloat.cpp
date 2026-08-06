// This test verifies that free function kernels don't bloat host object files.
// The __sycl_kernel_* wrapper should not have a body during host compilation.
// This addresses CMPLRLLVM-77222 where free function kernels caused ~3x bloat.
//
// Compile to object file (bundled)
// REQUIRES: linux
// RUN: %clangxx -fsycl -fsycl-targets=spir64 -c %s -o %t.o
//
// Extract host object
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu -input=%t.o -output=%t_host.o -unbundle
//
// Dump symbols to a file and verify wrapper is NOT defined
// RUN: llvm-nm %t_host.o > %t_host_syms.txt
//
// The wrapper function __sycl_kernel_* should not have type 'T' (defined text).
// If it appears with 'T', the fix has regressed.
// RUN: FileCheck %s --input-file %t_host_syms.txt --check-prefix=CHECK-HOST
//
// CHECK-HOST-NOT: {{[0-9a-f]+ T .*__sycl_kernel.*test_kernel}}

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// Free function kernel with nd_range_kernel property
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void test_kernel(sycl::nd_item<1> item, int *data, int n) {
  auto id = item.get_global_id(0);
  if (id < n) {
    data[id] = data[id] * 2 + 1;
  }
}

// Single task free function kernel
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void init_kernel(int *ptr, int value) {
  *ptr = value;
}

int main() {
  sycl::queue q;
  const int N = 64;

  // Allocate USM memory
  int *data = sycl::malloc_shared<int>(N, q);

  // Initialize data using single_task free function
  syclexp::single_task(q, syclexp::kernel_function_s<init_kernel>{}, data, 42);

  // Get kernel bundle for free function kernels
  auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q.get_context(), {q.get_device()},
      {sycl::get_kernel_id<syclexp::kernel_function_s<test_kernel>>()});

  // Launch free function kernel using nd_launch with kernel bundle
  q.submit([&](sycl::handler &h) {
    h.use_kernel_bundle(kb);
    sycl::nd_range<1> range{N, 64};
    syclexp::nd_launch(h, range, syclexp::kernel_function_s<test_kernel>{},
                       data, N);
  });

  // Alternative: direct queue submission
  sycl::nd_range<1> range{N, 64};
  syclexp::nd_launch(q, range, syclexp::kernel_function_s<test_kernel>{},
                     data, N);

  q.wait();

  sycl::free(data, q);
  return 0;
}
