// RUN: %{build} -o %t.out 
// RUN: %t.out
//

//==--------------- AMX_aspect.cpp - SYCL device test
//------------------------==//
//
// Checks that the has(aspect) method on a device returns the correct answer when
// queried about ext_intel_matrix AMX aspect.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using arch = sycl::ext::oneapi::experimental::architecture;
int main() {
  const std::vector<arch> supported_archs = {
      arch::intel_cpu_spr, arch::intel_gpu_pvc, arch::intel_gpu_dg2_g10,
      arch::intel_gpu_dg2_g11, arch::intel_gpu_dg2_g12};
  for (const auto &plt : platform::get_platforms()) {
    for (auto &dev : plt.get_devices()) {
      if (std::any_of(supported_archs.begin(), supported_archs.end(),
                      [&](const auto &a) {
                        return dev.ext_oneapi_architecture_is(a);
                      })) {
        assert(dev.has(sycl::aspect::ext_intel_matrix));
      }
    }
  }
  return 0;
}
