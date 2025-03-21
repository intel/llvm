// REQUIRES: gpu, level_zero

// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=MODE1 %s
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=MODE2 %s
// UNSUPPORTED: ze_debug, level_zero_v2_adapter

// Checks that with L0 device-scope events enabled the only host-visible L0
// event created is at the end of all kernels submission, when host waits for
// the last kernel's event.
//
// clang-format off
// MODE1-LABEL: Submitted all kernels
// MODE1: ---> urEventWait
// MODE1: ze_event_pool_desc_t flags set to: 1
// MODE1: zeEventCreate
// MODE1: zeCommandListAppendWaitOnEvents
// MODE1-NEXT: zeCommandListAppendSignalEvent
// MODE1: Completed all kernels

// With the SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 mode look for pattern that
// creates host-visible event just before command-list submission.
//
// MODE2: ze_event_pool_desc_t flags set to: 1
// MODE2: zeEventCreate
// MODE2: zeCommandListAppendBarrier
// MODE2: zeCommandListClose
// MODE2: zeCommandQueueExecuteCommandLists
// clang-format on

#include <iostream>
#include <sycl/detail/core.hpp>

int main(int argc, char **argv) {
  sycl::queue queue(sycl::gpu_selector_v);

  int N = (argc >= 2 ? std::atoi(argv[1]) : 100);
  std::cout << N << " kernels" << std::endl;

  sycl::event e; // completed event
  for (int i = 0; i < N; i++) {
    e = queue.submit([&](sycl::handler &h) {
      h.depends_on(e);
      h.single_task<class kernel>([=] {});
    });
  } // for

  std::cout << "Submitted all kernels" << std::endl;
  e.wait(); // Waits for the last kernel to complete.
  std::cout << "Completed all kernels" << std::endl;
  return 0;
}
