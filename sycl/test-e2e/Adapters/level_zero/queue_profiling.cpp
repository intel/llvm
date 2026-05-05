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

// Check the expected output when queue::enable_profiling is specified
//
// WITH: ze_event_pool_desc_t flags set to: 5
// clang-format on
//

#include <cassert>
#include <cstring>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
using namespace sycl;

void foo(queue &q, int n, bool profiling_enabled) {
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
      (void)startk;
      (void)endk;
      assert(profiling_enabled &&
             "Expected exception when profiling is not enabled");
    } catch (const sycl::exception &e) {
      assert(!profiling_enabled &&
             "Unexpected exception when profiling is enabled");
      const char *expected_msg =
          "Profiling information is unavailable as the "
          "queue associated with the event does not have "
          "the 'enable_profiling' property.";
      assert(std::strcmp(e.what(), expected_msg) == 0 &&
             "Exception message does not match expected message");
    }
  }
}

int main(int argc, char **argv) {

  bool profiling = argc > 1;

  {
    property_list propList{};
    if (profiling)
      propList = sycl::property::queue::enable_profiling();

    queue q(gpu_selector_v, propList);
    // Perform the computation
    foo(q, 2, profiling);
  } // SYCL scope

  return 0;
}
