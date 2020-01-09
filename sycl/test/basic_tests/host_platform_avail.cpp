// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==------ host_platform_avail.cpp - Host Platform Availability test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

class foo;
int main() {
  queue q;
  auto dev = q.get_device();
  auto dev_platform = dev.get_platform();

  if (!dev_platform.is_host()) {
    auto plats = platform::get_platforms();
    bool found_host = false;

    // Look for a host platform
    for (const auto &plat : plats)  {
      if (plat.is_host()) {
        found_host = true;
      }
    }
    // Fail if we didn't find a host platform
    if (!found_host)
      return 1;
  }

  // Found a host platform, all is well
  return 0;
}
