// REQUIRES: gpu, level_zero

// RUN: %{build}  -o %t.out
// RUN: env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-ENABLED %s
// RUN: env SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=0 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-ENABLED %s
// RUN: env SYCL_PI_LEVEL_ZERO_DISABLE_EVENTS_CACHING=1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=CACHING-DISABLED %s

// With events caching we should be reusing them and 9 should be enough.
// Might require more than one if previous one hasn't been released by the time
// we need a new one.

// CACHING-ENABLED: zeEventCreate = {{[1-9]}}
// CACHING-DISABLED: zeEventCreate = 256

// Check event caching modes in the L0 plugin.

#include <CL/sycl.hpp>

int main() {
  cl::sycl::queue deviceQueue;

  for (int i = 0; i < 256; i++) {
    auto Event = deviceQueue.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<class SimpleKernel>([=]() {});
    });
    Event.wait();
  }
  return 0;
}
