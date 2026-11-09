// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{ env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out %}

// Test: a signal enqueued after a host task is ordered after the host task, and
// a host task can signal the semaphore directly.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <chrono>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue producer{sycl::property::queue::in_order{}};
  sycl::queue consumer{producer.get_context(), producer.get_device(),
                       sycl::property::queue::in_order{}};
  syclex::timeline_semaphore sem{producer.get_context()};

  // Each check has its own input, so that the second producer host task does
  // not race with the first consumer kernel.
  int *values = sycl::malloc_shared<int>(2, producer);
  int *result = sycl::malloc_shared<int>(2, producer);
  values[0] = 0;
  values[1] = 0;

  producer.submit([&](sycl::handler &cgh) {
    cgh.host_task([=] {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      values[0] = 5;
    });
  });
  syclex::enqueue_signal(producer, sem, 1);

  syclex::enqueue_wait(consumer, sem, 1);
  consumer.single_task([=] { result[0] = values[0]; });

  producer.submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() mutable {
      values[1] = 7;
      sem.signal(2);
    });
  });

  syclex::enqueue_wait(consumer, sem, 2);
  consumer.single_task([=] { result[1] = values[1]; });

  consumer.wait();
  producer.wait();

  assert(result[0] == 5);
  assert(result[1] == 7);

  sycl::free(values, producer);
  sycl::free(result, producer);
  return 0;
}
