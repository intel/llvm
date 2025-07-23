//==--- opencl_multi_device.cpp --- kernel_compiler extension tests --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero) && ocloc

// RUN: %{build} -o %t.out
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=3 %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/platform.hpp>

// Test to check that bundle is buildable from OpenCL source if there are
// multiple devices in the context.

auto constexpr CLSource = R"===(
__kernel void Kernel1(int in, __global int *out) {
  out[0] = in;
}

__kernel void Kernel2(short in, __global short *out) {
  out[0] = in;
}
)===";

int main() {
  sycl::platform Platform;
  auto Context = Platform.khr_get_default_context();

  {
    auto devices = Context.get_devices();
    sycl::device d = devices[0];
    assert(d.ext_oneapi_cl_profile() != std::string{});
  }

  auto SourceKB =
      sycl::ext::oneapi::experimental::create_kernel_bundle_from_source(
          Context, sycl::ext::oneapi::experimental::source_language::opencl,
          CLSource);
  auto ExecKB = sycl::ext::oneapi::experimental::build(SourceKB);
  return 0;
}
