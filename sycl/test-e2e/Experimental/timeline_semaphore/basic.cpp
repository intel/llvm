// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: a consumer queue waits for a value that a producer queue signals.

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

  int *data = sycl::malloc_shared<int>(N, producer);

  producer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { data[i] = static_cast<int>(i); });
  syclex::enqueue_signal(producer, sem, 1);

  syclex::enqueue_wait(consumer, sem, 1);
  consumer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { data[i] += 1; });
  consumer.wait();

  // The value was reached, so this returns immediately.
  sem.wait(1);

  for (size_t i = 0; i < N; ++i)
    assert(data[i] == static_cast<int>(i) + 1);

  sycl::free(data, producer);
  return 0;
}
