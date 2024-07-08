// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -DVAR=1 -O2 -g -o %t1
// RUN: env SYCL_PREFER_UR=1 %{run} not %t1 2>&1 | FileCheck --check-prefixes CHECK,CHECK-VAR1 %s
// RUN: %{build} %device_asan_flags -DVAR=2 -O2 -g -o %t2
// RUN: env SYCL_PREFER_UR=1 %{run} not %t2 2>&1 | FileCheck --check-prefixes CHECK,CHECK-VAR2 %s
// RUN: %{build} %device_asan_flags -DVAR=3 -O2 -g -o %t3
// RUN: env SYCL_PREFER_UR=1 %{run} not %t3 2>&1 | FileCheck --check-prefixes CHECK,CHECK-VAR3 %s
// RUN: %{build} %device_asan_flags -DVAR=4 -O2 -g -o %t4
// RUN: env SYCL_PREFER_UR=1 %{run} not %t4 2>&1 | FileCheck --check-prefixes CHECK,CHECK-VAR4 %s
// RUN: %{build} %device_asan_flags -DVAR=5 -O2 -g -o %t5
// RUN: env SYCL_PREFER_UR=1 %{run} not %t5 2>&1 | FileCheck --check-prefixes CHECK,CHECK-VAR5 %s
// RUN: %{build} %device_asan_flags -DVAR=6 -O2 -g -o %t6
// RUN: env SYCL_PREFER_UR=1 %{run} not %t6 2>&1 | FileCheck --check-prefixes CHECK,CHECK-VAR6 %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

// CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Private Memory
template <typename T> __attribute__((noinline)) T foo(T *p) { return *p; }
template <typename T> __attribute__((noinline)) T foo1(T *p) { return *p; }
// CHECK-VAR1: READ of size 2 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID(0, 0, 0)
// CHECK-VAR1:   #0 {{.*}} {{.*multiple_private.cpp}}:[[@LINE-2]]
template <typename T> __attribute__((noinline)) T foo2(T *p) { return *p; }
// CHECK-VAR2: READ of size 2 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID(0, 0, 0)
// CHECK-VAR2:   #0 {{.*}} {{.*multiple_private.cpp}}:[[@LINE-2]]
template <typename T> __attribute__((noinline)) T foo3(T *p) { return *p; }
// CHECK-VAR3: READ of size 4 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID(0, 0, 0)
// CHECK-VAR3:   #0 {{.*}} {{.*multiple_private.cpp}}:[[@LINE-2]]
template <typename T> __attribute__((noinline)) T foo4(T *p) { return *p; }
// CHECK-VAR4: READ of size 4 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID(0, 0, 0)
// CHECK-VAR4:   #0 {{.*}} {{.*multiple_private.cpp}}:[[@LINE-2]]
template <typename T> __attribute__((noinline)) T foo5(T *p) { return *p; }
// CHECK-VAR5: READ of size 8 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID(0, 0, 0)
// CHECK-VAR5:   #0 {{.*}} {{.*multiple_private.cpp}}:[[@LINE-2]]
template <typename T> __attribute__((noinline)) T foo6(T *p) { return *p; }
// CHECK-VAR6: READ of size 1 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID(0, 0, 0)
// CHECK-VAR6:   #0 {{.*}} {{.*multiple_private.cpp}}:[[@LINE-2]]

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<long>(5, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() {
      short p1[] = {1};
      int p2[] = {1};
      int p3[10] = {8, 1, 10, 1, 0, 10};
      long p4[] = {5111LL};
      char p5[] = {'c'};

      array[0] = foo(&p1[0]);
      array[1] = foo(&p2[0]);
      for (int i = 0; i < 10; ++i)
        array[2] += foo(&p3[i]);
      array[3] = foo(&p4[0]);
      array[4] = foo(&p5[0]);

#if VAR == 1
      array[0] = foo1(&p1[-4]);
#elif VAR == 2
      array[0] = foo2(&p1[4]);
#elif VAR == 3
      array[0] = foo3(&p2[1]);
#elif VAR == 4
      array[0] = foo4(&p3[10]);
#elif VAR == 5
      array[0] = foo5(&p4[1]);
#else
      array[0] = foo6(&p5[1]);
#endif
    });
  });
  Q.wait();
  sycl::free(array, Q);

  return 0;
}
