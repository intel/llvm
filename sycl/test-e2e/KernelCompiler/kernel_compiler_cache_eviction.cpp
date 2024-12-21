//==-kernel_compiler_cache_eviction.cpp -- kernel_compiler extension tests -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests on-disk cache and eviction with kernel_compiler.

// REQUIRES: ocloc && (opencl || level_zero)
// UNSUPPORTED: accelerator

// -- Test the kernel_compiler with OpenCL source.
// RUN: %{build} -o %t.out

// -- Test again, with caching.
// DEFINE: %{cache_vars} = env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_TRACE=7 SYCL_CACHE_DIR=%t/cache_dir SYCL_CACHE_MAX_SIZE=23000
// RUN: rm -rf %t/cache_dir
// RUN: %{cache_vars} %t.out 2>&1 | FileCheck %s --check-prefix=CHECK

// CHECK: [Persistent Cache]: enabled

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

auto constexpr CLSource = R"===(
__kernel void my_kernel(__global int *in, __global int *out) {
  size_t i = get_global_id(0);
  out[i] = in[i]*2 + 100;
}
__kernel void her_kernel(__global int *in, __global int *out) {
  size_t i = get_global_id(0);
  out[i] = in[i]*5 + 1000;
}
)===";

using namespace sycl;

void test_build_and_run() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;

  // only one device is supported at this time, so we limit the queue and
  // context to that
  sycl::device d{sycl::default_selector_v};
  sycl::context ctx{d};
  sycl::queue q{ctx, d};

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::opencl);
  if (!ok) {
    std::cout << "Apparently this device does not support OpenCL C source "
                 "kernel bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return;
  }

  auto CreateAndVerifyKB = [](source_kb &kbSrc,
                              std::vector<std::string> &&BuildFlags) {
    std::string log;
    std::vector<sycl::device> devs = kbSrc.get_devices();
    sycl::context ctxRes = kbSrc.get_context();
    sycl::backend beRes = kbSrc.get_backend();

    auto kb =
        syclex::build(kbSrc, devs,
                      syclex::properties{syclex::build_options{BuildFlags},
                                         syclex::save_log{&log}});

    bool hasMyKernel = kb.ext_oneapi_has_kernel("my_kernel");
    bool hasHerKernel = kb.ext_oneapi_has_kernel("her_kernel");
    bool notExistKernel = kb.ext_oneapi_has_kernel("not_exist");
    assert(hasMyKernel && "my_kernel should exist, but doesn't");
    assert(hasHerKernel && "her_kernel should exist, but doesn't");
    assert(!notExistKernel && "non-existing kernel should NOT exist.");
  };

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, CLSource);

  // compilation with props and devices
  std::vector<std::string> flags{"-cl-fast-relaxed-math",
                                 "-cl-finite-math-only"};

  // Device image #1
  // CHECK: [Persistent Cache]: Cache size file not present. Creating one.
  // CHECK-NEXT: [Persistent Cache]: Cache size file created.
  // CHECK-NEXT: [kernel_compiler Persistent Cache]: binary has been cached: [[DEVIMG1:.*]]
  // CHECK-NEXT: [Persistent Cache]: Updating the cache size file.
  CreateAndVerifyKB(kbSrc, {});

  // Device image #2
  // CHECK-NEXT: [kernel_compiler Persistent Cache]: binary has been cached: [[DEVIMG2:.*]]
  // CHECK-NEXT: [Persistent Cache]: Updating the cache size file.
  CreateAndVerifyKB(kbSrc, {flags[0]});

  // Device image #3
  // CHECK: [kernel_compiler Persistent Cache]: binary has been cached: [[DEVIMG3:.*]]
  // CHECK: [Persistent Cache]: Updating the cache size file.
  CreateAndVerifyKB(kbSrc, {flags[1]});

  // Re-insert device image #1
  // CHECK: [kernel_compiler Persistent Cache]: using cached binary: [[DEVIMG1]]
  CreateAndVerifyKB(kbSrc, {});

  // Device image #4
  // CHECK: [kernel_compiler Persistent Cache]: binary has been cached: [[DEVIMG4:.*]]
  // CHECK: [Persistent Cache]: Updating the cache size file.
  // CHECK: [Persistent Cache]: Cache eviction triggered.
  // CHECK: [Persistent Cache]: File removed: [[DEVIMG2]]
  // CHECK: [Persistent Cache]: File removed: [[DEVIMG3]]
  // CHECK: [Persistent Cache]: File removed: [[DEVIMG1]]
  CreateAndVerifyKB(kbSrc, {flags[0], flags[1]});

  // Re-insert device image #4
  // CHECK: [kernel_compiler Persistent Cache]: using cached binary: [[DEVIMG4]]
  CreateAndVerifyKB(kbSrc, {flags[0], flags[1]});
}

int main() {
#ifndef SYCL_EXT_ONEAPI_KERNEL_COMPILER_OPENCL
  static_assert(false, "KernelCompiler OpenCL feature test macro undefined");
#endif

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  test_build_and_run();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
