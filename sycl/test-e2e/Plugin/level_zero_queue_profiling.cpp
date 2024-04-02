// REQUIRES: gpu, level_zero
// UNSUPPORTED: ze_debug

// RUN: %{build} -o %t.out
// RUN: env UR_L0_DEBUG=-1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=WITHOUT %s
// RUN: env UR_L0_DEBUG=-1 %{l0_leak_check} %{run} %t.out profile 2>&1 | FileCheck --check-prefixes=WITH %s

// Test case adapted from the SYCL version of Rodinia benchmark hotspot.

// clang-format off
// Check the expected output when queue::enable_profiling is not specified
//
// WITHOUT: ze_event_pool_desc_t flags set to: 1
// WITHOUT: SYCL exception caught: Profiling information is unavailable as the queue associated with the event does not have the 'enable_profiling' property.

// Check the expected output when queue::enable_profiling is specified
//
// WITH: ze_event_pool_desc_t flags set to: 5
// WITH: Device kernel time:
// clang-format on
//

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int foo(queue &q, int n) {
  for (int i = 0; i < n; i++) {

    sycl::event queue_event = q.submit([&](handler &cgh) {
      cgh.parallel_for<class empty>(range<2>(10000, 10000),
                                    [=](item<2> item) {});
    });

    q.wait();

    // Get kernel computation time
    try {
      auto startk = queue_event.template get_profiling_info<
          sycl::info::event_profiling::command_start>();
      auto endk = queue_event.template get_profiling_info<
          sycl::info::event_profiling::command_end>();
      auto kernel_time =
          (float)(endk - startk) * 1e-9f; // to seconds, 1e-6f to milliseconds
      printf("Device kernel time: %.12fs\n", (float)kernel_time);

    } catch (const sycl::exception &e) {
      std::cout << "SYCL exception caught: " << e.what() << '\n';
      return 0;
    }
  }
  return n;
}

int main(int argc, char **argv) {

  bool profiling = argc > 1;

  {
    property_list propList{};
    if (profiling)
      propList = sycl::property::queue::enable_profiling();

    queue q(gpu_selector_v, propList);
    // Perform the computation
    foo(q, 2);
  } // SYCL scope

  return 0;
}
