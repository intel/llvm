// REQUIRES: cpu
// UNSUPPORTED: windows
// RUN: %{build} -o %t.out
// RUN: %{run} sycl-trace --sycl  %t.out | FileCheck %s

// Test tracing of the code location data for queue.parallel_for in case of
// failure (exception generation)

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue Queue;
  sycl::device Dev = Queue.get_device();
  sycl::range<1> MaxWISizes =
      Dev.get_info<sycl::info::device::max_work_item_sizes<1>>();
  bool ExceptionCaught = false;
  try {
    // CHECK: code_location_queue_parallel_for.cpp:[[# @LINE + 3 ]] E2ETestKernel
    Queue.parallel_for<class E2ETestKernel>(
        sycl::nd_range<1>{MaxWISizes.get(0), 2 * MaxWISizes.get(0)},
        [](sycl::nd_item<1>) {});
  } catch (...) {
    ExceptionCaught = true;
  }
  Queue.wait();
  return !ExceptionCaught;
}
