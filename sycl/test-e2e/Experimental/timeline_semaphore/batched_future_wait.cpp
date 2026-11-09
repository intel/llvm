// REQUIRES: aspect-ext_oneapi_timeline_semaphore
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{ env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out %}

// Test: many waits and many ordinary commands can be submitted behind a wait
// that the submitting thread only satisfies afterwards. Submission must not
// block on the completion of earlier batches of a batched queue.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/timeline_semaphore.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>

namespace syclex = sycl::ext::oneapi::experimental;

// More waits than the number of batches that a batched queue keeps before it
// reclaims them, and more kernels than the number of commands in one batch.
constexpr int NumWaits = 25;
constexpr int NumKernels = 300;

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  syclex::timeline_semaphore sem{q.get_context()};

  int *counter = sycl::malloc_shared<int>(1, q);
  *counter = 0;

  for (int i = 0; i < NumWaits; ++i)
    syclex::enqueue_wait(q, sem, 1);
  for (int i = 0; i < NumKernels; ++i)
    q.single_task([=] { *counter += 1; });
  syclex::enqueue_signal(q, sem, 2);

  sem.signal(1);
  sem.wait(2);
  assert(*counter == NumKernels);

  q.wait();
  sycl::free(counter, q);
  return 0;
}
