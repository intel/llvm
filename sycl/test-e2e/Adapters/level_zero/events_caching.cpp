// REQUIRES: gpu, level_zero
// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-INTENDED: v2 adapter does not allow disabling caching

// RUN: %{build}  -o %t.out

// RUN: %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-ENABLED %s
// RUN: env SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-ENABLED %s
// RUN: env SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-DISABLED %s
//
// With events caching we should be reusing them and 9 should be enough.
// Might require more than one if previous one hasn't been released by the time
// we need a new one.

// CACHING-ENABLED: zeEventCreate = {{[1-9]}}
// CACHING-DISABLED: zeEventCreate = 256

// Check event caching modes in the L0 adapter.

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue deviceQueue;

  for (int i = 0; i < 256; i++) {
    auto Event = deviceQueue.submit([&](sycl::handler &cgh) {
      cgh.single_task<class SimpleKernel>([=]() {});
    });
    Event.wait();
  }
  return 0;
}
