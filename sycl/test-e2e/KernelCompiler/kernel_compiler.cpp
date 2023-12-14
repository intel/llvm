//==- kernel_compiler.cpp --- kernel_compiler extension tests  -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// the new spec for the kernel_compiler opens the door to supporting several
// different source languages. But, initially, OpenCL Kernels are the only ones
// supported. This test is limited to that (thus the cm-compiler requirement)
// but in the future it may need to broken out into other tests.

#include <sycl/sycl.hpp>

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

auto constexpr BadCLSource = R"===(
__kernel void my_kernel(__global int *in, __global int *out) {
  size_t i = get_global_id(0) +  no semi-colon!!
  out[i] = in[i]*2 + 100;
}
)===";
/*
Compile Log:
1:3:34: error: use of undeclared identifier 'no'
  size_t i = get_global_id(0) +  no semi-colon!!
                                 ^
1:3:36: error: expected ';' at end of declaration
  size_t i = get_global_id(0) +  no semi-colon!!
                                   ^
                                   ;

Build failed with error code: -11

=============

*/

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

  bool ok = syclex::is_source_kernel_bundle_supported(
      ctx.get_backend(), syclex::source_language::opencl);
  if (!ok) {
    std::cout << "Apparently this backend does not support OpenCL C source "
                 "kernel bundle extension: "
              << ctx.get_backend() << std::endl;
    return;
  }

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, CLSource);
  // compilation of empty prop list, no devices
  exe_kb kbExe1 = syclex::build(kbSrc);

  // compilation with props and devices
  std::string log;
  std::vector<std::string> flags{"-cl-fast-relaxed-math",
                                 "-cl-finite-math-only"};
  std::vector<sycl::device> devs = kbSrc.get_devices();
  sycl::context ctxRes = kbSrc.get_context();
  assert(ctxRes == ctx);
  sycl::backend beRes = kbSrc.get_backend();
  assert(beRes == ctx.get_backend());

  exe_kb kbExe2 = syclex::build(
      kbSrc, devs,
      syclex::properties{syclex::build_options{flags}, syclex::save_log{&log}});

  bool hasMyKernel = kbExe2.ext_oneapi_has_kernel("my_kernel");
  bool hasHerKernel = kbExe2.ext_oneapi_has_kernel("her_kernel");
  bool notExistKernel = kbExe2.ext_oneapi_has_kernel("not_exist");
  assert(hasMyKernel && "my_kernel should exist, but doesn't");
  assert(hasHerKernel && "her_kernel should exist, but doesn't");
  assert(!notExistKernel && "non-existing kernel should NOT exist, but does?");

  sycl::kernel my_kernel = kbExe2.ext_oneapi_get_kernel("my_kernel");
  sycl::kernel her_kernel = kbExe2.ext_oneapi_get_kernel("her_kernel");

  auto my_num_args = my_kernel.get_info<sycl::info::kernel::num_args>();
  assert(my_num_args == 2 && "my_kernel should take 2 args");

  testSyclKernel(q, my_kernel, 2, 100);
  testSyclKernel(q, her_kernel, 5, 1000);
}

void test_error() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  // only one device is supported at this time, so we limit the queue and
  // context to that
  sycl::device d{sycl::default_selector_v};
  sycl::context ctx{d};
  sycl::queue q{ctx, d};

  bool ok = syclex::is_source_kernel_bundle_supported(
      ctx.get_backend(), syclex::source_language::opencl);
  if (!ok) {
    return;
  }

  try {
    source_kb kbSrc = syclex::create_kernel_bundle_from_source(
        ctx, syclex::source_language::opencl, BadCLSource);
    exe_kb kbExe1 = syclex::build(kbSrc);
    assert(false && "we should not be here.");
  } catch (sycl::exception &e) {
    // nice!
    assert(e.code() == sycl::errc::build);
  }
  // any other error will escape and cause the test to fail ( as it should ).
}

int main() {
#ifndef SYCL_EXT_ONEAPI_KERNEL_COMPILER_OPENCL
  static_assert(false, "KernelCompiler OpenCL feature test macro undefined");
#endif

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  test_build_and_run();
  test_error();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
