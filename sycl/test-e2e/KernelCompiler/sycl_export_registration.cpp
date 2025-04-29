//==--- sycl_export_registration.cpp --- kernel_compiler extension tests ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// -- Test for a case where a kernel bundle is built that exports a symbol and
// -- other kernel bundles that uses it are compiled/linked without it. These
// -- cases should fail due to unresolved symbols, rather than picking up the
// -- symbol from the registered exported symbols.

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

auto constexpr SYCLExportSource = R"===(
#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = 0;
}

)===";

int main() {
  sycl::queue Q;
  int Failed = 0;

  if (!Q.get_device().ext_oneapi_can_build(syclex::source_language::sycl)) {
    std::cout << "Device does not support one of the source languages: "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return 0;
  }
  syclex::properties BuildOpts{
      syclex::build_options{"-fsycl-allow-device-image-dependencies"}};

  source_kb ImportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLImportSource);
  source_kb ExportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLExportSource);

  // Build the SYCL source that exports symbols.
  syclex::build(ExportSourceKB, BuildOpts);

  // Build the SYCL source that imports symbols separately. This should fail to
  // resolve exported symbols.
  bool BuildFailed = false;
  try {
    syclex::build(ImportSourceKB, BuildOpts);
  } catch (...) {
    BuildFailed = true;
  }
  if (!BuildFailed) {
    std::cout << "Building the SYCL source code with unresolved imported "
                 "symbols did NOT fail."
              << std::endl;
    ++Failed;
  }

  // Compiling the import kernel bundle should work, despite unresolved symbols.
  obj_kb ImportObjKB = syclex::compile(ImportSourceKB, BuildOpts);

  // Link the SYCL source that imports symbols separately. This should fail to
  // resolve exported symbols.
  bool LinkingFailed = false;
  try {
    sycl::link({ImportObjKB});
  } catch (...) {
    LinkingFailed = true;
  }
  if (!LinkingFailed) {
    std::cout << "Linking the SYCL source code with unresolved imported "
                 "symbols did NOT fail."
              << std::endl;
    ++Failed;
  }

  return Failed;
}
