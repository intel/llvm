//==----- kernel_compiler_sycl_jit_time_trace.cpp --- time-tracing test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: SYCL-RTC is not available for accelerator devices

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

int test_tracing() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;

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

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl_jit, "");

  auto props = syclex::properties{
      syclex::build_options{std::vector<std::string>{
          "-ftime-trace=-", "-ftime-trace-granularity=1000" /* us */,
          "-ftime-trace-verbose"}},
  };

  syclex::build(kbSrc, props);
  // CHECK: {"traceEvents":

  std::string log;
  auto props2 = syclex::properties{
      syclex::build_options{std::vector<std::string>{
          "-ftime-trace=-", "-ftime-trace-granularity=invalid_int"}},
      syclex::save_log{&log}};
  syclex::build(kbSrc, props2);
  std::cout << log << std::endl;
  // CHECK: {"traceEvents":
  // CHECK: warning: ignoring malformed argument: '-ftime-trace-granularity=invalid_int'

  return 0;
}

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  return test_tracing();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
