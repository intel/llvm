// Tests tracing of in-memory kernel and program cache.

// RUN: %{build} -o %t.out

// There should be no tracing output when SYCL_CACHE_IN_MEM is not set
// or SYCL_CACHE_TRACE is set to 0.

// RUN: env SYCL_CACHE_IN_MEM=0 %{run} %t.out 2> %t.trace1
// RUN: FileCheck --allow-empty --input-file=%t.trace1 --implicit-check-not "In-Memory Cache" %s
// RUN: env SYCL_CACHE_TRACE=0 %{run} %t.out 2> %t.trace2
// RUN: FileCheck --allow-empty --input-file=%t.trace2 --implicit-check-not "In-Memory Cache" %s

// RUN: env SYCL_CACHE_TRACE=2 %{run} %t.out 2> %t.trace3
// RUN: FileCheck %s --input-file=%t.trace3 --check-prefix=CHECK-CACHE-TRACE

#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

constexpr specialization_id<int> spec_id;

int main() {
  queue q;

  // Check program insertion into cache and kernel insertion into fast and
  // regular kernel cache.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:{{.*}}]: Program inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 0][Key:{Name = [[KERNELNAME1:.*]]]: Kernel inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 1][Key:{Name = [[KERNELNAME1]]]: Kernel inserted.

  // In the 2nd and 3rd invocation of this loop, the kernel should be fetched
  // from fast kernel cache.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 1][Key:{Name = [[KERNELNAME1]]]: Kernel fetched.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 1][Key:{Name = [[KERNELNAME1]]]: Kernel fetched.
  for (int i = 0; i < 3; i++)
    q.single_task([] {}).wait();

  auto *p = malloc_device<int>(1, q);

  // Check program and kernel insertion into cache. There should be different
  // programs for different iterations of this loop, because of the different
  // specialization constants.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:{{.*}}]: Program inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 0][Key:{Name = [[KERNELNAME2:.*]]]: Kernel inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Program Cache][Key:{{.*}}]: Program inserted.
  // CHECK-CACHE-TRACE: [In-Memory Cache][Thread Id:{{.*}}][Kernel Cache][IsFastCache: 0][Key:{Name = [[KERNELNAME2]]]: Kernel inserted.
  for (int i = 0; i < 2; ++i)
    q.submit([&](handler &cgh) {
       cgh.set_specialization_constant<spec_id>(i);
       cgh.parallel_for(1, [=](auto, kernel_handler kh) {
         *p = kh.get_specialization_constant<spec_id>();
       });
     }).wait();

  free(p, q);
}
