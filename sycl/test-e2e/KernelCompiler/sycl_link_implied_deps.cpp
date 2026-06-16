//==-- sycl_link_implied_deps.cpp - kernel_compiler implication test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// -- Regression test for CMPLRLLVM-75983: syclex::compile of an
// -- ext_oneapi_source bundle should imply
// -- -fsycl-allow-device-image-dependencies, since the resulting bundle is in
// -- bundle_state::object and is intended to be passed to sycl::link. The
// -- user should not have to repeat the flag via build_options for the
// -- runtime cross-image link to find the SYCL/exported and SYCL/imported
// -- symbol property sets.

// Note linking is not supported on CUDA/HIP.

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
auto constexpr SYCLImportSource = R"===(
#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr, int Size) {
  TestFunc(Ptr, Size);
}

)===";

auto constexpr SYCLExportSource = R"===(
#include <cstddef>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = I;
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

  source_kb ImportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLImportSource);
  source_kb ExportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLExportSource);

  // Intentionally do NOT pass
  // build_options{"-fsycl-allow-device-image-dependencies"}: the runtime
  // is expected to imply it for the object-state target of compile().
  obj_kb ImportObjKB = syclex::compile(ImportSourceKB);
  obj_kb ExportObjKB = syclex::compile(ExportSourceKB);

  exe_kb ExecKB = sycl::link({ImportObjKB, ExportObjKB});

  sycl::kernel Kernel = ExecKB.ext_oneapi_get_kernel("TestKernel");

  constexpr int Range = 10;
  int *USMPtr = sycl::malloc_shared<int>(Range, Q);

  memset(USMPtr, 0, Range * sizeof(int));
  Q.submit([&](sycl::handler &Handler) {
    Handler.set_args(USMPtr, Range);
    Handler.single_task(Kernel);
  });
  Q.wait();

  int Failed = 0;
  for (size_t I = 0; I < Range; ++I) {
    if (USMPtr[I] != static_cast<int>(I)) {
      std::cout << "Unexpected value at index " << I << ": " << USMPtr[I]
                << " != " << I << std::endl;
      ++Failed;
    }
  }

  sycl::free(USMPtr, Q);

  return Failed;
}
