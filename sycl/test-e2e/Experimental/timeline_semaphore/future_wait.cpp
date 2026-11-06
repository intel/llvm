// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: waits are enqueued before the work that signals the values they wait
// for, and a wait is satisfied by a larger value than its target.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 1024;

int main() {
  sycl::queue producer{sycl::property::queue::in_order{}};
  sycl::queue consumer{producer.get_context(), producer.get_device(),
                       sycl::property::queue::in_order{}};
  syclex::timeline_semaphore sem{producer.get_context()};

  int *first = sycl::malloc_shared<int>(N, producer);
  int *second = sycl::malloc_shared<int>(N, producer);
  int *result = sycl::malloc_shared<int>(N, producer);

  // The consumer is submitted first. Its second wait targets value 2, which
  // the producer reaches by signaling 3.
  syclex::enqueue_wait(consumer, sem, 1);
  consumer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { result[i] = first[i]; });
  syclex::enqueue_wait(consumer, sem, 2);
  consumer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { result[i] += second[i]; });

  producer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { first[i] = 1; });
  syclex::enqueue_signal(producer, sem, 1);
  producer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { second[i] = 10; });
  syclex::enqueue_signal(producer, sem, 3);

  consumer.wait();
  producer.wait();

  for (size_t i = 0; i < N; ++i)
    assert(result[i] == 11);

  sycl::free(first, producer);
  sycl::free(second, producer);
  sycl::free(result, producer);
  return 0;
}
