// REQUIRES: cpu
// UNSUPPORTED: windows
// RUN: %{build} -o %t.out
// RUN: %{run} sycl-trace --sycl  %t.out | FileCheck %s

// Test tracing of the code location data for queue.parallel_for in case of
// failure (exception generation)

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Queue;
  sycl::buffer<int, 1> Buf(8);
  sycl::device Dev = Queue.get_device();
  sycl::range<1> MaxWISizes =
      Dev.get_info<sycl::info::device::max_work_item_sizes<1>>();
  bool ExceptionCaught = false;
  try {
// CHECK: code_location_queue_parallel_for.cpp:[[# @LINE + 3 ]] E2ETestKernel
    Queue.parallel_for<class E2ETestKernel>(
        sycl::nd_range<1>{MaxWISizes.get(0), 2 * MaxWISizes.get(0)},
        [](sycl::id<1> idx) {});
  } catch (...) {
    ExceptionCaught = true;
  }
  Queue.wait();
  return !ExceptionCaught;
}
