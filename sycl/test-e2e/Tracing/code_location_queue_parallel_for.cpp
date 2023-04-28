// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER sycl-trace --sycl  %t.out %CPU_CHECK_PLACEHOLDER

// Test tracing of the code location data for queue.parallel_for in case of
// failure (exception generation)
//
// CHECK: code_location_queue_parallel_for.cpp:22 E2ETestKernel

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Queue;
  sycl::buffer<int, 1> Buf(8);
  sycl::device Dev = Queue.get_device();
  sycl::id<1> MaxWISizes =
      Dev.get_info<sycl::info::device::max_work_item_sizes<1>>();
  bool ExceptionCaught = false;
  try {
    Queue.parallel_for<class E2ETestKernel>(
        sycl::nd_range<1>{MaxWISizes.get(0), 2 * MaxWISizes.get(0)},
        [](sycl::id<1> idx) {});
  } catch (...) {
    ExceptionCaught = true;
  }
  Queue.wait();
  return !ExceptionCaught;
}
