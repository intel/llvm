// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// There is an issue with reported device time for the L0 backend, works only on
// pvc for now. No such problems for opencl backend.
// REQUIRES: !ext_oneapi_level_zero || gpu-intel-pvc

// Check that submission time is calculated properly.

#include <sycl/sycl.hpp>

using namespace sycl;

int main(void) {
  queue q({property::queue::enable_profiling{}});
  int *data = malloc_host<int>(1024, q);

  for (int i = 0; i < 20; i++) {
    auto event = q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class KernelTime>(sycl::range<1>(1024),
                                         [=](id<1> idx) { data[idx] = idx; });
    });

    event.wait();
    auto submit_time =
        event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();

    if (!(submit_time <= start_time) || !(start_time <= end_time))
      return -1;
  }
  return 0;
}
