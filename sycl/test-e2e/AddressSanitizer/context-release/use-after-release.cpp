// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: env UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:1" %{run} not --crash %t.out freed kernel 2>&1 | FileCheck %s --check-prefixes=CHECK-ARG,CHECK-FREED
// RUN: env UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:1" %{run} not --crash %t.out leaked kernel 2>&1 | FileCheck %s --check-prefixes=CHECK-LEAK,CHECK-ARG,CHECK-LEAKED
// RUN: env UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:0" %{run} not --crash %t.out freed kernel 2>&1 | FileCheck %s --check-prefixes=CHECK-DEV,CHECK-FREED
// RUN: env UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:0" %{run} not --crash %t.out leaked kernel 2>&1 | FileCheck %s --check-prefixes=CHECK-LEAK,CHECK-DEV,CHECK-LEAKED
// RUN: %{run} not --crash %t.out freed free 2>&1 | FileCheck %s --check-prefixes=CHECK-DF,CHECK-DF-FREED
// RUN: %{run} not --crash %t.out leaked free 2>&1 | FileCheck %s --check-prefixes=CHECK-LEAK,CHECK-DF,CHECK-DF-LEAKED

// Device USM of a released context is used in another context, after the
// program freed it ("freed") or not ("leaked"). Releasing the context releases
// the memory, so both must be reported as a use of released memory.

#include <cstring>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main(int, char **argv) {
  bool Freed = !std::strcmp(argv[1], "freed");
  bool UseInKernel = !std::strcmp(argv[2], "kernel");
  sycl::device Dev;
  int *P = nullptr;
  {
    sycl::context Ctx1{Dev};
    sycl::queue Q1{Ctx1, Dev};
    P = sycl::malloc_device<int>(1, Q1);
    if (Freed)
      sycl::free(P, Q1);
  }

  sycl::context Ctx2{Dev};
  sycl::queue Q2{Ctx2, Dev};
  if (UseInKernel)
    Q2.single_task<class MyKernel>([=]() { *P = 0; }).wait();
  else
    sycl::free(P, Q2);
  return 0;
}

// CHECK-LEAK: ERROR: DeviceSanitizer: detected memory leaks of Device USM

// CHECK-ARG: ERROR: DeviceSanitizer: invalid-argument on kernel <{{.*MyKernel}}>
// CHECK-ARG: The {{[0-9]+}}th argument [[ADDR:0x[0-9a-f]+]] is a released USM pointer
// CHECK-DEV: ERROR: DeviceSanitizer: use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK-DEV: WRITE of size 4 at kernel <{{.*MyKernel}}>
// CHECK-DEV-NOT: Failed to find
// CHECK-ARG: [[ADDR]] is located inside of Device USM region
// CHECK-DEV: [[ADDR]] is located inside of Device USM region
// CHECK-FREED: allocated here:
// CHECK-FREED: in main {{.*use-after-release.cpp:}}[[@LINE-24]]
// CHECK-FREED: freed here:
// CHECK-FREED: in main {{.*use-after-release.cpp:}}[[@LINE-24]]
// CHECK-LEAKED: allocated here:
// CHECK-LEAKED: in main {{.*use-after-release.cpp:}}[[@LINE-28]]
// CHECK-LEAKED: freed here:
// CHECK-LEAKED: urContextRelease

// CHECK-DF: ERROR: DeviceSanitizer: double-free on address [[DFADDR:0x[0-9a-f]+]]
// CHECK-DF: [[DFADDR]] is located inside of Device USM region
// CHECK-DF: freed here:
// CHECK-DF-FREED: in main {{.*use-after-release.cpp:}}[[@LINE-33]]
// CHECK-DF-LEAKED: urContextRelease
// CHECK-DF: previously allocated here:
// CHECK-DF: in main {{.*use-after-release.cpp:}}[[@LINE-38]]
