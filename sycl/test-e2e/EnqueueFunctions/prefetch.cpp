// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

// Tests the enqueue free function prefetch.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/usm.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;
constexpr size_t ChunkSize = N / 3;

int main() {
  sycl::context Context;
  sycl::queue Q(Context, sycl::default_selector_v);
  int *Memory = sycl::malloc_shared<int>(N, Q);

  oneapiext::prefetch(Q, Memory, ChunkSize);

  oneapiext::submit(Q, [&](sycl::handler &CGH) {
    oneapiext::prefetch(CGH, Memory + ChunkSize, ChunkSize);
  });

  sycl::event E = oneapiext::submit_with_event(Q, [&](sycl::handler &CGH) {
    oneapiext::prefetch(CGH, Memory + ChunkSize * 2, ChunkSize);
  });

  E.wait();
  Q.wait();
  sycl::free(Memory, Q);

  return 0;
}

// CHECK-COUNT-3:---> piextUSMEnqueuePrefetch
// CHECK-NOT:---> piextUSMEnqueuePrefetch
