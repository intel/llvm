// RUN: %{build} -I . -o %t.out
// RUN: %{run} %t.out

#include <cstdlib>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>
#include <sycl/properties/all_properties.hpp>

int main() {
  sycl::queue q{{sycl::property::queue::enable_profiling()}};

  auto e = q.submit([&](sycl::handler &cgh) { cgh.host_task([=]() {}); });
  q.wait();

  const uint64_t submitted = e.template get_profiling_info<
      sycl::info::event_profiling::command_submit>();
  const uint64_t start = e.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  const uint64_t end =
      e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  if (submitted == 0) {
    std::cerr << "Invalid command_submit time" << std::endl;
    return EXIT_FAILURE;
  }

  if (start < submitted) {
    std::cerr << "Invalid command_start time" << std::endl;
    return EXIT_FAILURE;
  }

  if (end < start) {
    std::cerr << "Invalid command_end time" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
