// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test to check that it is possible to get profiling info from the event
// returned by barrier which turns into NOP.

#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>

int main() {
  sycl::event start;
  sycl::event stop;
  sycl::queue q{sycl::property_list(sycl::property::queue::in_order(),
                                    sycl::property::queue::enable_profiling())};
  float elapsed = 0;

  start = q.ext_oneapi_submit_barrier();
  std::cout << "before parallel_for" << std::endl;
  q.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 16),
                        sycl::range<3>(1, 1, 16)),
      [=](sycl::nd_item<3> item_ct1) {
        double d = 123;
        for (int i = 0; i < 10000; i++) {
          d = d * i;
        }
      });
  std::cout << "after parallel_for" << std::endl;
  stop = q.ext_oneapi_submit_barrier();
  stop.wait_and_throw();
  elapsed =
      (stop.get_profiling_info<sycl::info::event_profiling::command_end>() -
       start.get_profiling_info<sycl::info::event_profiling::command_start>()) /
      1000000.0f;
  std::cout << "elapsed:" << elapsed << std::endl;
  return 0;
}
