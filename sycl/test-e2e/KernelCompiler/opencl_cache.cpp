//==--- opencl.cpp --- kernel_compiler extension tests ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc && (opencl || level_zero)

// -- Test the kernel_compiler with OpenCL source.
// RUN: %{build} -o %t.out

// -- Test with caching.
// DEFINE: %{cache_vars} = env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_TRACE=5 SYCL_CACHE_DIR=%t/cache_dir
// RUN: %{run-aux} rm -rf %t/cache_dir
// RUN: %{l0_leak_check} %{cache_vars} %{run} %t.out 2>&1 |  FileCheck %s --check-prefixes=CHECK-WRITTEN-TO-CACHE
// RUN: %{l0_leak_check} %{cache_vars} %{run} %t.out 2>&1 |  FileCheck %s --check-prefixes=CHECK-READ-FROM-CACHE

// CHECK-WRITTEN-TO-CACHE: [Persistent Cache]: enabled
// CHECK-WRITTEN-TO-CACHE-NOT: [kernel_compiler Persistent Cache]: using cached binary
// CHECK-WRITTEN-TO-CACHE: [kernel_compiler Persistent Cache]: binary has been cached

// CHECK-READ-FROM-CACHE: [Persistent Cache]: enabled
// CHECK-READ-FROM-CACHE-NOT: [kernel_compiler Persistent Cache]: binary has been cached
// CHECK-READ-FROM-CACHE: [kernel_compiler Persistent Cache]: using cached binary

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

void testSyclKernel(sycl::queue &Q, sycl::kernel Kernel, int multiplier,
                    int added) {
  constexpr int N = 4;
  cl_int InputArray[N] = {0, 1, 2, 3};
  cl_int OutputArray[N] = {};

  sycl::buffer InputBuf(InputArray, sycl::range<1>(N));
  sycl::buffer OutputBuf(OutputArray, sycl::range<1>(N));

  Q.submit([&](sycl::handler &CGH) {
    CGH.set_arg(0, InputBuf.get_access<sycl::access::mode::read>(CGH));
    CGH.set_arg(1, OutputBuf.get_access<sycl::access::mode::write>(CGH));
    CGH.parallel_for(sycl::range<1>{N}, Kernel);
  });

  sycl::host_accessor Out{OutputBuf};
  for (int I = 0; I < N; I++)
    assert(Out[I] == ((I * multiplier) + added));
}

void test_build_and_run() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  // only one device is supported at this time, so we limit the queue and
  // context to that
  sycl::device d{sycl::default_selector_v};
  sycl::context ctx{d};
  sycl::queue q{ctx, d};

  bool ok =
      q.get_device().ext_oneapi_can_build(syclex::source_language::opencl);
  if (!ok) {
    std::cout << "Apparently this device does not support OpenCL C source "
                 "kernel bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return;
  }

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, CLSource);
  exe_kb kbExe1 = syclex::build(kbSrc);

  sycl::kernel my_kernel = kbExe1.ext_oneapi_get_kernel("my_kernel");

  testSyclKernel(q, my_kernel, 2, 100);
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
