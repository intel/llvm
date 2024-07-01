// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int foo(int p[], int i) { return p[i]; }
// CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Private Memory
// CHECK: READ of size 4 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID({{.*}}, 0, 0)
// CHECK:   #0 {{.*}} {{.*single_private.cpp}}:[[@LINE-3]]

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() {
      int p[] = {1, 2, 3, 4};
      for (int i = 0; i < 5; ++i)
        array[0] = foo(p, i);
    });
  });
  Q.wait();
  sycl::free(array, Q);

  return 0;
}
