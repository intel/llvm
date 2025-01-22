//==- sycl_and_cache.cpp - cache works with kernel_compiler sycl support
//----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// -- Test the kernel_compiler with SYCL source.
// RUN: %{build} -o %t.out

// -- Run with caching.

// DEFINE: %{cache_vars} = %{l0_leak_check} env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_TRACE=5 SYCL_CACHE_DIR=%t/cache_dir
// RUN: rm -rf %t/cache_dir
// RUN:  %{cache_vars} %t.out 2>&1 |  FileCheck %s --check-prefixes=CHECK-WRITTEN-TO-CACHE
// RUN:  %{cache_vars} %t.out 2>&1 |  FileCheck %s --check-prefixes=CHECK-READ-FROM-CACHE

// CHECK-WRITTEN-TO-CACHE: [Persistent Cache]: enabled
// CHECK-WRITTEN-TO-CACHE-NOT: [kernel_compiler Persistent Cache]: using cached binary
// CHECK-WRITTEN-TO-CACHE: [kernel_compiler Persistent Cache]: binary has been cached

// CHECK-READ-FROM-CACHE: [Persistent Cache]: enabled
// CHECK-READ-FROM-CACHE-NOT: [kernel_compiler Persistent Cache]: binary has been cached
// CHECK-READ-FROM-CACHE: [kernel_compiler Persistent Cache]: using cached binary

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

// TODO: remove SYCL_EXTERNAL once it is no longer needed.
auto constexpr SYCLSource = R"===(
#include <sycl/sycl.hpp>

int AddEm(int a, int b){
    return a + b + 5;
}

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_cp(int *ptr) {

  // intentionally using deprecated routine, as opposed to this_work_item::get_nd_item<1>()
  sycl::nd_item<1> Item = sycl::ext::oneapi::experimental::this_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = AddEm(GId.get(0), 37);
}
)===";

void test_1(sycl::queue &Queue, sycl::kernel &Kernel, int seed) {
  constexpr int Range = 10;
  int *usmPtr = sycl::malloc_shared<int>(Range, Queue);
  int start = 3;

  sycl::nd_range<1> R1{{Range}, {1}};

  bool Passa = true;

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](sycl::handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.parallel_for(R1, Kernel);
  });
  Queue.wait();

  for (int i = 0; i < Range; i++) {
    std::cout << usmPtr[i] << "=" << (i + seed) << " ";
    assert(usmPtr[i] == i + seed);
  }
  std::cout << std::endl;

  sycl::free(usmPtr, Queue);
}

void test_build_and_run() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl);
  if (!ok) {
    std::cout << "Apparently this device does not support SYCL source "
                 "kernel bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return;
  }

  // Create from source.
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource);

  // Compilation of empty prop list, no devices.
  exe_kb kbExe = syclex::build(kbSrc);

  // extern "C" was used, so the name "ff_cp" is not mangled and can be used
  // directly.
  sycl::kernel k = kbExe.ext_oneapi_get_kernel("ff_cp");

  // Test the kernels.
  test_1(q, k, 37 + 5); // ff_cp seeds 37. AddEm will add 5 more.
}

int main(int argc, char *argv[]) {

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  test_build_and_run();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
