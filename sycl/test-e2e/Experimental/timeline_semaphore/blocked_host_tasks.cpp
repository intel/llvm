// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{ env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out %}

// Test: signal and wait operations enqueued behind host tasks that block until
// the main thread releases them, so the operations are held by the SYCL runtime
// when they are submitted.
//
// The host tasks may share a single host-task thread, so no host task waits for
// another one. The consumer host task is submitted first and must finish before
// the producer is released, which makes the wait reach the backend before the
// signal that satisfies it.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <atomic>
#include <cassert>
#include <thread>

namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 256;

static void waitFor(const std::atomic<bool> &Flag) {
  while (!Flag.load(std::memory_order_acquire))
    std::this_thread::yield();
}

// The producer writes the input in a blocked host task and signals 1 after it.
// The consumer waits for 1 behind its own blocked host task and then runs
// kernels that read the input.
static void testProducerConsumer(sycl::queue &producer, sycl::queue &consumer) {
  syclex::timeline_semaphore sem{producer.get_context()};
  int *input = sycl::malloc_shared<int>(N, producer);
  int *first = sycl::malloc_shared<int>(N, producer);
  int *second = sycl::malloc_shared<int>(N, producer);
  for (size_t i = 0; i < N; ++i)
    input[i] = 0;

  std::atomic<bool> releaseConsumer{false};
  std::atomic<bool> consumerDone{false};
  std::atomic<bool> releaseProducer{false};

  consumer.submit([&](sycl::handler &cgh) {
    cgh.host_task([&] {
      waitFor(releaseConsumer);
      consumerDone.store(true, std::memory_order_release);
    });
  });
  syclex::enqueue_wait(consumer, sem, 1);
  consumer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { first[i] = input[i]; });
  consumer.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { second[i] = input[i] * 2; });

  producer.submit([&](sycl::handler &cgh) {
    cgh.host_task([&] {
      waitFor(releaseProducer);
      for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<int>(i) + 1;
    });
  });
  syclex::enqueue_signal(producer, sem, 1);

  releaseConsumer.store(true, std::memory_order_release);
  waitFor(consumerDone);
  releaseProducer.store(true, std::memory_order_release);

  consumer.wait();
  producer.wait();

  for (size_t i = 0; i < N; ++i) {
    assert(first[i] == static_cast<int>(i) + 1);
    assert(second[i] == 2 * (static_cast<int>(i) + 1));
  }

  sycl::free(input, producer);
  sycl::free(first, producer);
  sycl::free(second, producer);
}

// The host signals the value while the wait for it is still held behind a
// blocked host task.
static void testHostSignalWhileWaitHeld(sycl::queue &q) {
  syclex::timeline_semaphore sem{q.get_context()};
  int *input = sycl::malloc_shared<int>(1, q);
  int *output = sycl::malloc_shared<int>(1, q);
  *input = 0;
  *output = 0;

  std::atomic<bool> release{false};
  q.submit(
      [&](sycl::handler &cgh) { cgh.host_task([&] { waitFor(release); }); });
  syclex::enqueue_wait(q, sem, 1);
  q.single_task([=] { *output = *input + 1; });

  *input = 41;
  sem.signal(1);
  release.store(true, std::memory_order_release);

  q.wait();
  assert(*output == 42);

  sycl::free(input, q);
  sycl::free(output, q);
}

int main() {
  {
    sycl::queue producer{sycl::property::queue::in_order{}};
    sycl::queue consumer{producer.get_context(), producer.get_device(),
                         sycl::property::queue::in_order{}};
    testProducerConsumer(producer, consumer);
  }
  {
    sycl::queue producer;
    sycl::queue consumer{producer.get_context(), producer.get_device()};
    testProducerConsumer(producer, consumer);
  }
  {
    sycl::queue q{sycl::property::queue::in_order{}};
    testHostSignalWhileWaitHeld(q);
  }
  return 0;
}
