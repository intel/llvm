// REQUIRES: aspect-usm_device_allocations
// RUN: %{build} %threads_lib -o %t.out
// RUN: %{run} %t.out

// Regression test for a case where parallel work with enqueue functions
// discarding their results would cause implicit waits on discarded events.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>
#include <thread>

void threadFunction(int) {
  sycl::queue Q{{sycl::property::queue::in_order()}};

  constexpr int Size = 128 * 128 * 128;
  int *DevMem = sycl::malloc_device<int>(Size, Q);

  sycl::ext::oneapi::experimental::submit(
      Q, [&](sycl::handler &cgh) { cgh.fill<int>(DevMem, 1, Size); });
  Q.wait_and_throw();

  sycl::free(DevMem, Q);
}

int main() {
  constexpr size_t NThreads = 2;
  std::array<std::thread, NThreads> Threads;

  for (size_t I = 0; I < NThreads; I++)
    Threads[I] = std::thread{threadFunction, I};
  for (size_t I = 0; I < NThreads; I++)
    Threads[I].join();

  return 0;
}
