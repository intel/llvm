//==--- sycl_split_barrier.cpp --- kernel_compiler split barrier test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: sycl-jit, gpu, level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The kernel_compiler (sycl-jit) translates the kernel to SPIR-V with its own
// extension allow-list. Check that SPV_INTEL_split_barrier is allowed there:
// without it the build below fails with "RequiresExtension". The kernel is
// only built, not run, so the test does not depend on the GPU running split
// barriers.

#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;

  std::string source = R"""(
    #include <sycl/ext/oneapi/free_function_queries.hpp>
    #include <sycl/ext/oneapi/kernel_properties.hpp>
    namespace syclext = sycl::ext::oneapi;
    namespace syclexp = sycl::ext::oneapi::experimental;

    extern "C" SYCL_EXTERNAL void __spirv_ControlBarrierArriveINTEL(int, int,
                                                                   int);
    extern "C" SYCL_EXTERNAL void __spirv_ControlBarrierWaitINTEL(int, int,
                                                                 int);

    extern "C"
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
    void split_barrier(int *ptr) {
      size_t id =
          syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
      // Scope Workgroup (2); Release/Acquire | WorkgroupMemory.
      __spirv_ControlBarrierArriveINTEL(2, 2, 0x108);
      __spirv_ControlBarrierWaitINTEL(2, 2, 0x104);
      ptr[id] = static_cast<int>(id);
    }
  )""";

  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclexp::create_kernel_bundle_from_source(
          q.get_context(), syclexp::source_language::sycl, source);
  sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
      syclexp::build(kb_src);
  if (!kb_exe.ext_oneapi_has_kernel("split_barrier")) {
    std::cout << "Kernel split_barrier not found\n";
    return 1;
  }
  return 0;
}
