// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test: device work waits for a value that the host signals, and the host
// waits for a value that the device signals.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  syclex::timeline_semaphore sem{q.get_context(), /*initial_value*/ 10};

  int *input = sycl::malloc_shared<int>(1, q);
  int *output = sycl::malloc_shared<int>(1, q);
  *input = 0;
  *output = 0;

  // A value below the initial value is reached already.
  syclex::enqueue_wait(q, sem, 5);
  syclex::enqueue_wait(q, sem, 11);
  q.single_task([=] { *output = *input * 2; });
  syclex::enqueue_signal(q, sem, 12);

  // The kernel is gated until the host has written the input.
  *input = 21;
  sem.signal(11);

  sem.wait(12);
  assert(*output == 42);

  q.wait();
  sycl::free(input, q);
  sycl::free(output, q);
  return 0;
}
