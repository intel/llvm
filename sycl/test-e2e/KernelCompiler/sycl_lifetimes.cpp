//==--------- sycl_lifetimes.cpp - kernel_compiler lifetime tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true
// UNSUPPORTED-INTENDED: sycl-jit is disabled on this branch.

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=-1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

auto constexpr SYCLSource = R"""(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void vec_add(float* in1, float* in2, float* out){
  size_t id = sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
  out[id] = in1[id] + in2[id];
}
)""";

int test_lifetimes() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl);
  if (!ok) {
    std::cout << "Apparently this device does not support `sycl` source "
                 "kernel bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return -1;
  }

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource);

  exe_kb kbExe1 = syclex::build(kbSrc);
  // CHECK: urProgramCreateWithIL{{.*}}phProgram{{.*}}([[PROG1:.*]]))

  {
    std::cout << "Scope1\n";
    // CHECK: Scope1
    exe_kb kbExe2 = syclex::build(kbSrc);
    // kbExe2 goes out of scope; its kernels are removed from program mananager.
    // CHECK: urProgramCreateWithIL{{.*}}phProgram{{.*}}([[PROG2:.*]]))
    // CHECK: urProgramRelease{{.*}}[[PROG2]]
  }
  std::cout << "End Scope1\n";
  // CHECK: End Scope1

  {
    std::cout << "Scope2\n";
    // CHECK: Scope2
    std::unique_ptr<sycl::kernel> kPtr;
    {
      std::cout << "Scope3\n";
      // CHECK: Scope3
      exe_kb kbExe3 = syclex::build(kbSrc);

      sycl::kernel k = kbExe3.ext_oneapi_get_kernel("vec_add");
      // CHECK: urKernelCreate{{.*}}phKernel{{.*}}([[KERNEL1:.*]]))
      kPtr = std::make_unique<sycl::kernel>(k);
      // kbExe3 goes out of scope, but the kernel keeps the underlying
      // impl-object alive
      // CHECK-NOT: urKernelRelease
    }
    std::cout << "End Scope3\n";
    // CHECK: End Scope3
    // kPtr goes out of scope, freeing the kernel and its bundle
    // CHECK: urKernelRelease{{.*}}[[KERNEL1]]
  }
  std::cout << "End Scope2\n";
  // CHECK: End Scope2
  // CHECK: urProgramRelease{{.*}}[[PROG1]]

  return 0;
}

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  return test_lifetimes();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
