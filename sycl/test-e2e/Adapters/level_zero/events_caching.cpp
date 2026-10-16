// REQUIRES: gpu, level_zero

// RUN: %{build}  -o %t.out

// RUN: %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=%if level_zero_v2_adapter %{CHECK-V2%} %else %{CACHING-ENABLED%} %s
// RUN: %if !level_zero_v2_adapter %{env SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-ENABLED %s%}
// RUN: %if !level_zero_v2_adapter %{env SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-DISABLED %s%}
//
// With events caching we should be reusing them and 9 should be enough.
// Might require more than one if previous one hasn't been released by the time
// we need a new one.
//
// L0v2 has no on/off switch for event caching. Which internal API creates
// the event, and whether it's traced by UR_L0_LEAKS_DEBUG at all, varies by
// driver/OS, so check the property directly instead: count distinct native
// event handles returned across 256 operations. Reuse means far fewer than
// 256 distinct handles.

// CACHING-ENABLED: zeEventCreate = {{[1-9]}}
// CACHING-DISABLED: zeEventCreate = 256
// CHECK-V2: Distinct native event handles: {{[1-9][0-9]?}}

// Check event caching modes in the L0 adapter.

#include <iostream>
#include <set>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

int main() {
  sycl::queue deviceQueue;
  std::set<ze_event_handle_t> distinctEvents;

  for (int i = 0; i < 256; i++) {
    auto Event = deviceQueue.submit([&](sycl::handler &cgh) {
      cgh.single_task<class SimpleKernel>([=]() {});
    });
    Event.wait();
    distinctEvents.insert(
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Event));
  }
  std::cout << "Distinct native event handles: " << distinctEvents.size()
            << "\n";
  return 0;
}
