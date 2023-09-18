// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check that device_impl::isGetDeviceAndHostTimerSupported() is not spoiling
// device_impl::MDeviceHostBaseTime values used for submit timestamp
// calculation.

#include <sycl/sycl.hpp>

using namespace sycl;

int main(void) {
  sycl::queue queue(
      sycl::property_list{sycl::property::queue::enable_profiling()});
  sycl::event event = queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class set_value>(sycl::range<1>{1024},
                                      [=](sycl::id<1> idx) {});
  });

  // SYCL RT internally calls device_impl::isGetDeviceAndHostTimerSupported()
  // to decide how to calculate "submit" timestamp - either using backend API
  // call or using fallback implementation.
  auto submit =
      event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto start =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();

  if (!(submit <= start) || !(start <= end))
    return -1;

  return 0;
}
