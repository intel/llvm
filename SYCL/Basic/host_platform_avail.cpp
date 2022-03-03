// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: env SYCL_DEVICE_FILTER=acc,host %t1.out

// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero

//==------ host_platform_avail.cpp - Host Platform Availability test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  auto plats = platform::get_platforms();
  bool found_host = false;

  // Look for a host platform
  for (const auto &plat : plats) {
    if (plat.is_host()) {
      found_host = true;
    }
  }

  // Fail if we didn't find a host platform
  return (!found_host);
}
