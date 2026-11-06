// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: two semaphores drive a producer/consumer pipeline over several rounds
// that share one buffer, without resetting the semaphores.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdint>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 256;
constexpr uint64_t Rounds = 10;

int main() {
  sycl::queue producer{sycl::property::queue::in_order{}};
  sycl::queue consumer{producer.get_context(), producer.get_device(),
                       sycl::property::queue::in_order{}};
  sycl::context ctxt = producer.get_context();
  syclex::timeline_semaphore produced{ctxt};
  syclex::timeline_semaphore consumed{ctxt};

  int *buf = sycl::malloc_device<int>(N, producer);
  int *sums = sycl::malloc_shared<int>(Rounds, producer);
  for (uint64_t r = 0; r < Rounds; ++r)
    sums[r] = 0;

  for (uint64_t r = 1; r <= Rounds; ++r) {
    // Do not overwrite buf before the consumer has read round r - 1.
    syclex::enqueue_wait(producer, consumed, r - 1);
    producer.parallel_for(sycl::range<1>{N},
                          [=](sycl::id<1> i) { buf[i] = static_cast<int>(r); });
    syclex::enqueue_signal(producer, produced, r);

    syclex::enqueue_wait(consumer, produced, r);
    consumer.single_task([=] {
      int sum = 0;
      for (size_t i = 0; i < N; ++i)
        sum += buf[i];
      sums[r - 1] = sum;
    });
    syclex::enqueue_signal(consumer, consumed, r);
  }

  consumed.wait(Rounds);
  producer.wait();
  consumer.wait();

  for (uint64_t r = 1; r <= Rounds; ++r)
    assert(sums[r - 1] == static_cast<int>(r * N));

  sycl::free(buf, ctxt);
  sycl::free(sums, ctxt);
  return 0;
}
