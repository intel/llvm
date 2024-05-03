// RUN: %{build} -o %t.out
// There is an issue with reported device time for the L0 backend, works only on
// pvc for now. No such problems for other backends.
// RUN: %if (!level_zero || gpu-intel-pvc) %{ %{run} %t.out %}

// Check that submission time is calculated properly.

// Test fails on hip flakily, disable temprorarily.
// UNSUPPORTED: hip

#include <sycl/sycl.hpp>

int main(void) {
  sycl::queue q({sycl::property::queue::enable_profiling{}});
  int *data = malloc_host<int>(1024, q);

  for (int i = 0; i < 20; i++) {
    auto event = q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class KernelTime>(
          sycl::range<1>(1024), [=](sycl::id<1> idx) { data[idx] = idx; });
    });

    event.wait();
    auto submit_time =
        event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();

    assert((submit_time <= start_time) && (start_time <= end_time));
  }
  sycl::free(data, q);
  return 0;
}
