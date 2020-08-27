// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
//==------------------- ReleaseResourcesExplicitDep.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include "../helpers.hpp"

class KernelNameA;
class KernelNameB;

// Check that the resources are correctly released for command groups with only
// explicit dependencies.
int main() {
  // CHECK:---> piContextCreate
  // CHECK:---> piQueueCreate
  // CHECK:---> piProgramCreate

  // CHECK:---> piKernelCreate
  // CHECK:---> piEnqueueKernelLaunch
  sycl::queue Q{sycl::cpu_selector{}};
  sycl::event EventA = Q.single_task<KernelNameA>([]() {});

  // CHECK:---> piKernelCreate
  // CHECK:---> piEnqueueKernelLaunch
  sycl::event EventB = Q.single_task<KernelNameB>(EventA, []() {});
  EventB.wait();
}

// CHECK:---> piEventRelease
// CHECK:---> piEventRelease
// CHECK:---> piQueueRelease
// CHECK:---> piProgramRelease
// CHECK:---> piContextRelease
// CHECK:---> piKernelRelease
// CHECK:---> piKernelRelease
