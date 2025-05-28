// RUN: %{build} -o %t.out
// There is an issue with reported device time for the L0 backend, works only on
// pvc for now. No such problems for other backends.
// RUN: %if (!level_zero || arch-intel_gpu_pvc) %{ %{run} %t.out %}

// Check that submission time is calculated properly.

// Test fails on hip flakily, disable temprorarily.
// UNSUPPORTED: hip

#include <cassert>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

int main(void) {
  constexpr size_t n = 16;
  sycl::queue q({sycl::property::queue::enable_profiling{}});
  int *data = sycl::malloc_host<int>(n, q);
  int *dest = sycl::malloc_host<int>(n, q);

  // Large enough to expose incorrect submit time (exceeding start time).
  constexpr int iterations = 5000;
  for (int i = 0; i < iterations; i++) {
    auto event = q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class KernelTime>(
          sycl::range<1>(n), [=](sycl::id<1> idx) { data[idx] = idx; });
    });

    event.wait();
    auto submit_time =
        event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();

    // Print for debugging
    std::cout << "Kernel Event - Submit: " << submit_time
              << ", Start: " << start_time << ", End: " << end_time
              << std::endl;

    assert(submit_time != 0 && "Submit time should not be zero");
    assert((submit_time <= start_time) && (start_time <= end_time));
  }

  // All shortcut memory operations use queue_impl::submitMemOpHelper.
  // This test covers memcpy as a representative, extend if other operations
  // diverge.
  for (int i = 0; i < iterations; i++) {
    auto memcpy_event = q.memcpy(dest, data, sizeof(int) * n);
    memcpy_event.wait();

    auto submit_time =
        memcpy_event
            .get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        memcpy_event
            .get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        memcpy_event
            .get_profiling_info<sycl::info::event_profiling::command_end>();

    // Print for debugging
    std::cout << "Memcpy Event - Submit: " << submit_time
              << ", Start: " << start_time << ", End: " << end_time
              << std::endl;

    assert(submit_time != 0 && "Submit time should not be zero");
    assert((submit_time <= start_time) && (start_time <= end_time));
  }

  sycl::free(data, q);
  sycl::free(dest, q);
  return 0;
}
