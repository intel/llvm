// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -Xarch_device "-mllvm=-asan-spir-privates=1" -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=detect_privates:1 %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -Xarch_device "-mllvm=-asan-spir-privates=1" -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=detect_privates:1 %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -Xarch_device "-mllvm=-asan-spir-privates=1" -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=detect_privates:1 %{run} not %t 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

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
    Q.wait();
  });

  return 0;
}
