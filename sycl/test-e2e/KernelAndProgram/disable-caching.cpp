// This test ensures created program/kernels are not retained
// if and only if caching is disabled.

// RUN: %{build} -o %t.out
// RUN: env ZE_DEBUG=-6 SYCL_UR_TRACE=2 SYCL_CACHE_IN_MEM=0 %{run} %t.out \
// RUN: | FileCheck %s
// RUN: env ZE_DEBUG=-6 SYCL_UR_TRACE=2 %{run} %t.out \
// RUN: | FileCheck %s --check-prefixes=CHECK-CACHE

// TODO: Reenable on Windows, see https://github.com/intel/llvm/issues/14768
// XFAIL: windows

#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

constexpr specialization_id<int> spec_id;

int main() {
  queue q;
  // CHECK: urProgramCreate
  // CHECK-NOT: urProgramRetain
  // CHECK: urKernelCreate
  // CHECK-NOT: urKernelRetain
  // CHECK: urEnqueueKernelLaunch
  // CHECK: urKernelRelease
  // CHECK: urProgramRelease
  // CHECK: urEventWait

  // CHECK-CACHE: urProgramCreate
  // CHECK-CACHE: urProgramRetain
  // CHECK-CACHE-NOT: urProgramRetain
  // CHECK-CACHE: urKernelCreate
  // CHECK-CACHE: urKernelRetain
  // CHECK-CACHE-NOT: urKernelCreate
  // CHECK-CACHE: urEnqueueKernelLaunch
  // CHECK-CACHE: urKernelRelease
  // CHECK-CACHE: urProgramRelease
  // CHECK-CACHE: urEventWait
  q.single_task([] {}).wait();

  // CHECK: urProgramCreate
  // CHECK-NOT: urProgramRetain
  // CHECK: urKernelCreate
  // CHECK-NOT: urKernelRetain
  // CHECK: urEnqueueKernelLaunch
  // CHECK: urKernelRelease
  // CHECK: urProgramRelease
  // CHECK: urEventWait

  // CHECK-CACHE: urProgramCreate
  // CHECK-CACHE: urProgramRetain
  // CHECK-CACHE-NOT: urProgramRetain
  // CHECK-CACHE: urKernelCreate
  // CHECK-CACHE: urKernelRetain
  // CHECK-CACHE-NOT: urKernelCreate
  // CHECK-CACHE: urEnqueueKernelLaunch
  // CHECK-CACHE: urKernelRelease
  // CHECK-CACHE: urProgramRelease
  // CHECK-CACHE: urEventWait

  // CHECK: urProgramCreate
  // CHECK-NOT: urProgramRetain
  // CHECK: urKernelCreate
  // CHECK-NOT: urKernelRetain
  // CHECK: urEnqueueKernelLaunch
  // CHECK: urKernelRelease
  // CHECK: urProgramRelease
  // CHECK: urEventWait

  // CHECK-CACHE: urProgramCreate
  // CHECK-CACHE: urProgramRetain
  // CHECK-CACHE-NOT: urProgramRetain
  // CHECK-CACHE: urKernelCreate
  // CHECK-CACHE: urKernelRetain
  // CHECK-CACHE-NOT: urKernelCreate
  // CHECK-CACHE: urEnqueueKernelLaunch
  // CHECK-CACHE: urKernelRelease
  // CHECK-CACHE: urProgramRelease
  // CHECK-CACHE: urEventWait
  auto *p = malloc_device<int>(1, q);
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
// CHECK-CACHE: urKernelRelease
// CHECK-CACHE: urKernelRelease
// CHECK-CACHE: urKernelRelease
// CHECK-CACHE: urProgramRelease
// CHECK-CACHE: urProgramRelease
// CHECK-CACHE: urProgramRelease
