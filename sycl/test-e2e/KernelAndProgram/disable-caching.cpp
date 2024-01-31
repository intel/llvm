// REQUIRES: usm_shared_allocations
// This test ensures created program/kernels are not retained
// if and only if caching is disabled.

// RUN: %{build} -o %t.out
// RUN: env ZE_DEBUG=-6 SYCL_PI_TRACE=-1 SYCL_CACHE_IN_MEM=0 %{run} %t.out \
// RUN: | FileCheck %s
// RUN: env ZE_DEBUG=-6 SYCL_PI_TRACE=-1 %{run} %t.out \
// RUN: | FileCheck %s --check-prefixes=CHECK-CACHE
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr specialization_id<int> spec_id;

int main() {
  queue q;
  // CHECK: piProgramCreate
  // CHECK-NOT: piProgramRetain
  // CHECK: piKernelCreate
  // CHECK-NOT: piKernelRetain
  // CHECK: piEnqueueKernelLaunch
  // CHECK: piKernelRelease
  // CHECK: piProgramRelease
  // CHECK: piEventsWait

  // CHECK-CACHE: piProgramCreate
  // CHECK-CACHE: piProgramRetain
  // CHECK-CACHE-NOT: piProgramRetain
  // CHECK-CACHE: piKernelCreate
  // CHECK-CACHE: piKernelRetain
  // CHECK-CACHE-NOT: piKernelCreate
  // CHECK-CACHE: piEnqueueKernelLaunch
  // CHECK-CACHE: piKernelRelease
  // CHECK-CACHE: piProgramRelease
  // CHECK-CACHE: piEventsWait
  q.single_task([] {}).wait();

  // CHECK: piProgramCreate
  // CHECK-NOT: piProgramRetain
  // CHECK: piKernelCreate
  // CHECK-NOT: piKernelRetain
  // CHECK: piEnqueueKernelLaunch
  // CHECK: piKernelRelease
  // CHECK: piProgramRelease
  // CHECK: piEventsWait

  // CHECK-CACHE: piProgramCreate
  // CHECK-CACHE: piProgramRetain
  // CHECK-CACHE-NOT: piProgramRetain
  // CHECK-CACHE: piKernelCreate
  // CHECK-CACHE: piKernelRetain
  // CHECK-CACHE-NOT: piKernelCreate
  // CHECK-CACHE: piEnqueueKernelLaunch
  // CHECK-CACHE: piKernelRelease
  // CHECK-CACHE: piProgramRelease
  // CHECK-CACHE: piEventsWait

  // CHECK: piProgramCreate
  // CHECK-NOT: piProgramRetain
  // CHECK: piKernelCreate
  // CHECK-NOT: piKernelRetain
  // CHECK: piEnqueueKernelLaunch
  // CHECK: piKernelRelease
  // CHECK: piProgramRelease
  // CHECK: piEventsWait

  // CHECK-CACHE: piProgramCreate
  // CHECK-CACHE: piProgramRetain
  // CHECK-CACHE-NOT: piProgramRetain
  // CHECK-CACHE: piKernelCreate
  // CHECK-CACHE: piKernelRetain
  // CHECK-CACHE-NOT: piKernelCreate
  // CHECK-CACHE: piEnqueueKernelLaunch
  // CHECK-CACHE: piKernelRelease
  // CHECK-CACHE: piProgramRelease
  // CHECK-CACHE: piEventsWait
  auto *p = malloc_shared<int>(1, q);
  for (int i = 0; i < 2; ++i)
    q.submit([&](handler &cgh) {
       cgh.set_specialization_constant<spec_id>(i);
       cgh.parallel_for(1, [=](auto, kernel_handler kh) {
         *p = kh.get_specialization_constant<spec_id>();
       });
     }).wait();

  free(p, q);
}

// (Program cache releases)
// CHECK-CACHE: piKernelRelease
// CHECK-CACHE: piKernelRelease
// CHECK-CACHE: piKernelRelease
// CHECK-CACHE: piProgramRelease
// CHECK-CACHE: piProgramRelease
// CHECK-CACHE: piProgramRelease
