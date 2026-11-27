// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t1.out
// RUN: %force_device_asan_rt %{run} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -DLEAK -O0 -g -o %t2.out
// RUN: %force_device_asan_rt UR_LAYER_ASAN_OPTIONS=detect_leaks:0 %{run} %t2.out 2>&1 | FileCheck %s

// Releasing a context frees its memory, so a new context may get the same
// device USM addresses. Allocations there must not be reported as bad-context
// or double-free.

#include <cstdio>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::device Dev;
  for (int I = 0; I < 2; ++I) {
    sycl::context Ctx{Dev};
    sycl::queue Q{Ctx, Dev};
    int *P = sycl::malloc_device<int>(80, Q);
#ifndef LEAK
    sycl::free(P, Q);
#endif
  }
  std::printf("done\n");
  return 0;
}

// CHECK-NOT: ERROR: DeviceSanitizer
// CHECK: done
