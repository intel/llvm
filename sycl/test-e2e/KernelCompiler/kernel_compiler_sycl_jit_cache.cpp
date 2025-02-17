//==- kernel_compiler_sycl_jit_cache.cpp --- persistent cache for SYCL-RTC -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// DEFINE: %{cache_vars} = env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_TRACE=7 SYCL_CACHE_DIR=%t/cache_dir
// DEFINE: %{max_cache_size} = SYCL_CACHE_MAX_SIZE=30000
// RUN: %{build} -o %t.out
// RUN: %{run-aux} rm -rf %t/cache_dir
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-UNLIM
// RUN: %{run-aux} rm -rf %t/cache_dir
// RUN: %{cache_vars} %{max_cache_size} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-EVICT

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

auto constexpr SYCLSource = R"""(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void vec_add(float* in1, float* in2, float* out){
  size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
  out[id] = in1[id] + in2[id];
}
)""";

auto constexpr SYCLSourceWithInclude = R"""(
  #include "myheader.h"
  #include <sycl/sycl.hpp>
  
  extern "C" SYCL_EXTERNAL 
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
  void KERNEL_NAME(float* in1, float* out){
    size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
    out[id] = in1[id];
  }
  )""";

static void dumpKernelIDs() {
  for (auto &kernelID : sycl::get_kernel_ids())
    std::cout << kernelID.get_name() << std::endl;
}

int test_persistent_cache() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl_jit);
  if (!ok) {
    std::cout << "Apparently this device does not support `sycl_jit` source "
                 "kernel bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return -1;
  }

  source_kb kbSrc1 = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl_jit, SYCLSource);

  // Bundle is entered into cache on first build.
  // CHECK: [kernel_compiler Persistent Cache]: cache miss: [[KEY1:.*]]
  // CHECK: [kernel_compiler Persistent Cache]: storing device code IR: {{.*}}/[[KEY1]]
  exe_kb kbExe1a = syclex::build(kbSrc1);
  dumpKernelIDs();
  // CHECK: rtc_0$__sycl_kernel_vec_add

  // Cache hit! We get independent bundles with their own version of the kernel.
  // CHECK: [kernel_compiler Persistent Cache]: using cached device code IR: {{.*}}/[[KEY1]]
  exe_kb kbExe1b = syclex::build(kbSrc1);
  dumpKernelIDs();
  // CHECK-DAG: rtc_0$__sycl_kernel_vec_add
  // CHECK-DAG: rtc_1$__sycl_kernel_vec_add

  source_kb kbSrc2 = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl_jit, SYCLSource);

  // Different source bundle, but identical source is a cache hit.
  // CHECK: [kernel_compiler Persistent Cache]: using cached device code IR: {{.*}}/[[KEY1]]
  exe_kb kbExe2a = syclex::build(kbSrc2);

  // Different build_options means no cache hit.
  // CHECK: [kernel_compiler Persistent Cache]: cache miss: [[KEY2:.*]]
  // CHECK: [kernel_compiler Persistent Cache]: storing device code IR: {{.*}}/[[KEY2]]
  std::vector<std::string> flags{"-g", "-fno-fast-math"};
  exe_kb kbExe1c =
      syclex::build(kbSrc1, syclex::properties{syclex::build_options{flags}});

  // The kbExe1c build should trigger eviction if cache size is limited.
  // CHECK-UNLIM: [kernel_compiler Persistent Cache]: using cached device code IR: {{.*}}/[[KEY1]]
  // CHECK-EVICT: [Persistent Cache]: Cache eviction triggered.
  // CHECK-EVICT: [Persistent Cache]: File removed: {{.*}}/[[KEY1]]
  // CHECK-EVICT: [kernel_compiler Persistent Cache]: cache miss: [[KEY1]]
  // CHECK-EVICT: [kernel_compiler Persistent Cache]: storing device code IR: {{.*}}/[[KEY1]]
  exe_kb kbExe2b = syclex::build(kbSrc2);

  source_kb kbSrc3 = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl_jit, SYCLSourceWithInclude,
      syclex::properties{
          syclex::include_files{"myheader.h", "#define KERNEL_NAME foo"}});

  // New source string -> cache miss
  // CHECK: [kernel_compiler Persistent Cache]: cache miss: [[KEY3:.*]]
  // CHECK: [kernel_compiler Persistent Cache]: storing device code IR: {{.*}}/[[KEY3]]
  exe_kb kbExe3a = syclex::build(kbSrc3);
  dumpKernelIDs();
  // CHECK: rtc_5$__sycl_kernel_foo

  source_kb kbSrc4 = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl_jit, SYCLSourceWithInclude,
      syclex::properties{
          syclex::include_files{"myheader.h", "#define KERNEL_NAME bar"}});

  // Same source string, but different header contents -> cache miss
  // CHECK: [kernel_compiler Persistent Cache]: cache miss: [[KEY4:.*]]
  // CHECK: [kernel_compiler Persistent Cache]: storing device code IR: {{.*}}/[[KEY4]]
  exe_kb kbExe4a = syclex::build(kbSrc4);
  dumpKernelIDs();
  // CHECK: rtc_6$__sycl_kernel_bar

  return 0;
}

int main(int argc, char **) {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  return test_persistent_cache();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
