// RUN: %{build} -D__SYCL_INTERNAL_API -o %t.run
// RUN: %{run} %t.run

//==--- execution_capabilities_legacy.cpp - deprecated query behavior ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <sycl/sycl.hpp>

int main() {
  for (const auto &Plt : sycl::platform::get_platforms()) {
    for (const auto &Dev : Plt.get_devices()) {
      const bool IsOpenCL = Dev.get_backend() == sycl::backend::opencl;
      try {
        [[maybe_unused]] auto Caps =
            Dev.get_info<sycl::info::device::execution_capabilities>();
        assert(IsOpenCL &&
               "execution_capabilities is expected to work only on OpenCL");
      } catch (const sycl::exception &E) {
        assert(!IsOpenCL &&
               "OpenCL backend is not expected to throw for this query");
        assert(E.code() == sycl::errc::invalid &&
               "Non-OpenCL backends must throw errc::invalid");
      }
    }
  }

  return 0;
}
