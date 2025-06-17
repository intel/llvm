//==- sycl_link_common_dep_optional_feature.cpp --- kernel_compiler tests --==//
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

// -- Test for linking where two kernels use the same imported symbols, but one
// -- may not be supported on the device.

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
auto constexpr SYCLImportSource1 = R"===(
#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel1(int *Ptr, int Size) {
  TestFunc(Ptr, Size);
}

)===";

// TODO: remove SYCL_EXTERNAL from the kernel once it is no longer needed.
auto constexpr SYCLImportSource2 = R"===(
#include <sycl/sycl.hpp>

SYCL_EXTERNAL void TestFunc(int *Ptr, int Size);

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel2(int *Ptr, int Size) {
  TestFunc(Ptr, Size);
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = static_cast<int>(static_cast<double>(Ptr[I]) / 2.0);
}

)===";

auto constexpr SYCLExportSource = R"===(
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

  source_kb ImportSourceKB1 = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLImportSource1);
  source_kb ImportSourceKB2 = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLImportSource2);
  source_kb ExportSourceKB = syclex::create_kernel_bundle_from_source(
      Q.get_context(), syclex::source_language::sycl, SYCLExportSource);

  syclex::properties BuildOpts{
      syclex::build_options{"-fsycl-allow-device-image-dependencies"}};
  obj_kb ImportObjKB1 = syclex::compile(ImportSourceKB1, BuildOpts);
  obj_kb ImportObjKB2 = syclex::compile(ImportSourceKB2, BuildOpts);
  obj_kb ExportObjKB = syclex::compile(ExportSourceKB, BuildOpts);

  exe_kb ExecKB = sycl::link({ImportObjKB1, ImportObjKB2, ExportObjKB});

  constexpr int Range = 10;
  int *USMPtr = sycl::malloc_shared<int>(Range, Q);

  int Failed = 0;
  {
    sycl::kernel Kernel = ExecKB.ext_oneapi_get_kernel("TestKernel1");

    memset(USMPtr, 0, Range * sizeof(int));
    Q.submit([&](sycl::handler &Handler) {
      Handler.set_args(USMPtr, Range);
      Handler.single_task(Kernel);
    });
    Q.wait();

    for (size_t I = 0; I < Range; ++I) {
      if (USMPtr[I] != I) {
        std::cout << "TestKernel1: Unexpected value at index " << I << ": "
                  << USMPtr[I] << " != " << I << std::endl;
        ++Failed;
      }
    }
  }

  if (Q.get_device().has(sycl::aspect::fp64)) {
    sycl::kernel Kernel = ExecKB.ext_oneapi_get_kernel("TestKernel2");

    memset(USMPtr, 0, Range * sizeof(int));
    Q.submit([&](sycl::handler &Handler) {
      Handler.set_args(USMPtr, Range);
      Handler.single_task(Kernel);
    });
    Q.wait();

    for (size_t I = 0; I < Range; ++I) {
      const int Expected = static_cast<int>(static_cast<double>(I) / 2.0);
      if (USMPtr[I] != Expected) {
        std::cout << "TestKernel2: Unexpected value at index " << I << ": "
                  << USMPtr[I] << " != " << Expected << std::endl;
        ++Failed;
      }
    }
  } else if (ExecKB.ext_oneapi_has_kernel("TestKernel2")) {
    std::cout << "Device does not support fp64, but the kernel bundle still "
                 "has the kernel using it."
              << std::endl;
    ++Failed;
  }

  sycl::free(USMPtr, Q);

  return Failed;
}
