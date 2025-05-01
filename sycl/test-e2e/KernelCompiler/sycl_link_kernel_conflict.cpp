//==-- sycl_link_kernel_conflict.cpp --- kernel_compiler extension tests ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// -- Test for the linking of two kernels with conflicting definitions of
// -- kernels with the same name.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
using obj_kb = sycl::kernel_bundle<sycl::bundle_state::object>;
using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

// TODO: remove SYCL_EXTERNAL from the kernel once it is no longer needed.
auto constexpr SYCLSource1 = R"===(
#include <sycl/sycl.hpp>

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr) {
  *Ptr = 42;
}

)===";

// TODO: remove SYCL_EXTERNAL from the kernel once it is no longer needed.
auto constexpr SYCLSource2 = R"===(
#include <sycl/sycl.hpp>

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr) {
  *Ptr = 24;
}

)===";

int main() {
  sycl::queue Q;

  if (!Q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl)) {
    std::cout << "Device does not support one of the source languages: "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return 0;
  }

  source_kb SourceKB1 = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLSource1);
  source_kb SourceKB2 = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLSource2);

  obj_kb ObjKB1 = syclex::compile(SourceKB1);
  obj_kb ObjKB2 = syclex::compile(SourceKB2);

  try {
    sycl::link({ObjKB1, ObjKB2});
  } catch (sycl::exception &E) {
    std::cout << "Exception caught: " << E.what() << std::endl;
    return 0;
  }

  std::cout << "No exception caught while linking two binaries with "
               "conflicting kernels."
            << std::endl;

  return 1;
}
