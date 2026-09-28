// REQUIRES: level_zero_v2_adapter, usm

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/reusable_events.hpp>
#include <sycl/usm.hpp>

#include <cassert>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  constexpr size_t numElements = 1024;
  sycl::queue syclQueue;
  auto event = syclex::make_event(syclQueue.get_context(),
                                  syclex::properties{syclex::event_mode{
                                      syclex::event_mode_enum::low_power}});

  int *data = sycl::malloc_shared<int>(numElements, syclQueue);

  syclQueue.parallel_for<class LowPowerKernel>(
      sycl::range<1>(numElements), [=](sycl::id<1> idx) { data[idx] = 42; });
  syclex::enqueue_signal_event(syclQueue, event);
  event.wait();

  for (size_t i = 0; i < numElements; ++i)
    assert(data[i] == 42);

  sycl::free(data, syclQueue);
  return 0;
}
