// Tests eviction for in-memory program cache.

// Future optimizations in device code reduction can cause this test to fail.
// Therefore, adding O0.
// RUN: %{build} %O0 -o %t.out

// RUN: env SYCL_CACHE_TRACE=2 SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD=20000 %{run} %t.out 2> %t.trace3
// RUN: FileCheck %s --input-file=%t.trace3 --check-prefix=CHECK-CACHE-TRACE

#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

constexpr specialization_id<int> spec_id;

int main() {
  queue q;

  // The first time the kernel is used, it will be stored in kernel and fast
  // kernel cache. Then next two times it will be fetched from fast kernel
  // cache.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:[[KEY1:.*]]]: Program inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:[[KEY1]]]: Program added to the end of eviction list.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 0][Key:{Name = [[KERNELNAME1:.*]]]: Kernel inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 1][Key:{Name = [[KERNELNAME1:.*]]]: Kernel inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 1][Key:{Name = [[KERNELNAME1:.*]]]: Kernel fetched.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 1][Key:{Name = [[KERNELNAME1:.*]]]: Kernel fetched.
  for (int i = 0; i < 3; i++)
    q.single_task([] {}).wait();

  auto *p = malloc_device<int>(1, q);

  // Added program and kernel for first loop iteration.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:[[KEY2:.*]]]: Program inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:[[KEY2]]]: Program added to the end of eviction list.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 0][Key:{Name = [[KERNELNAME2:.*]]]: Kernel inserted.

  // Added program and kernel for second loop iteration.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:[[KEY3:.*]]]: Program inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:[[KEY3]]]: Program added to the end of eviction list.

  // Eviction triggered. The first program will be evicted from cache.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 0][Key:{Name = [[KERNELNAME1]]]: Kernel evicted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 1][Key:{Name = [[KERNELNAME1]]]: Kernel evicted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:[[KEY1]]]: Program evicted.

  // Kernel for second loop iteration will be inserted into kernel cache.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 0][Key:{Name = [[KERNELNAME3:.*]]]: Kernel inserted.

  for (int i = 0; i < 2; ++i)
    q.submit([&](handler &cgh) {
       cgh.set_specialization_constant<spec_id>(i);
       cgh.parallel_for(1, [=](auto, kernel_handler kh) {
         *p = kh.get_specialization_constant<spec_id>();
       });
     }).wait();

  free(p, q);
}
