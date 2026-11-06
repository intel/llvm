// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{ env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out %}

// Test: on out-of-order queues, enqueue_signal covers all previously submitted
// commands and enqueue_wait gates all subsequently submitted commands.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>
#include <sycl/usm.hpp>

#include <cassert>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue producer;
  sycl::queue consumer{producer.get_context(), producer.get_device()};
  syclex::timeline_semaphore sem{producer.get_context()};

  int *a = sycl::malloc_shared<int>(N, producer);
  int *b = sycl::malloc_shared<int>(N, producer);
  int *sum = sycl::malloc_shared<int>(N, producer);
  int *product = sycl::malloc_shared<int>(N, producer);

  // Two independent kernels, both covered by the signal.
  producer.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) { a[i] = 1; });
  producer.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) { b[i] = 2; });
  syclex::enqueue_signal(producer, sem, 1);

  // Two independent kernels, both gated by the wait. They only read the
  // producer outputs, so they do not depend on each other.
  syclex::enqueue_wait(consumer, sem, 1);
  consumer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { sum[i] = a[i] + b[i]; });
  consumer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { product[i] = b[i] * 10; });

  consumer.wait();
  producer.wait();

  for (size_t i = 0; i < N; ++i) {
    assert(sum[i] == 3);
    assert(product[i] == 20);
  }

  sycl::free(a, producer);
  sycl::free(b, producer);
  sycl::free(sum, producer);
  sycl::free(product, producer);
  return 0;
}
