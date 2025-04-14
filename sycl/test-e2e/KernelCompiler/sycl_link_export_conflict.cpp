//==-- sycl_link_export_conflict.cpp --- kernel_compiler extension tests ---==//
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

// -- Test for a case where a kernel bundle with an exported symbol is compiled
// -- before another kernel bundle using a different variant of the symbol.

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
#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr, int Size) {
  TestFunc(Ptr, Size);
}

)===";

auto constexpr SYCLExportSource1 = R"===(
#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = 0;
}

)===";

auto constexpr SYCLExportSource2 = R"===(
#include <sycl/sycl.hpp>

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
  syclex::properties BuildOpts{
      syclex::build_options{"-fsycl-allow-device-image-dependencies"}};

  source_kb ConflictExportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLExportSource1);
  exe_kb ConflictingExecKB = syclex::build(ConflictExportSourceKB, BuildOpts);

  source_kb ImportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLImportSource);
  source_kb ExportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLExportSource2);

  obj_kb ImportObjKB = syclex::compile(ImportSourceKB, BuildOpts);
  obj_kb ExportObjKB = syclex::compile(ExportSourceKB, BuildOpts);

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
    if (USMPtr[I] != I) {
      std::cout << "Unexpected value at index " << I << ": " << USMPtr[I]
                << " != " << I << std::endl;
      ++Failed;
    }
  }

  sycl::free(USMPtr, Q);

  return Failed;
}
