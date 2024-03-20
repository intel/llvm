// REQUIRES: gpu, level_zero

// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1 SYCL_PI_TRACE=-1 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=MODE1 %s
// RUN: env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_TRACE=-1 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --check-prefixes=MODE2 %s
// UNSUPPORTED: ze_debug

// Checks that with L0 device-scope events enabled the only host-visible L0
// event created is at the end of all kernels submission, when host waits for
// the last kernel's event.
//
// clang-format off
// MODE1-LABEL: Submitted all kernels
// MODE1: ---> piEventsWait(
// MODE1-NEXT:        <unknown> : 1
// MODE1: ze_event_pool_desc_t flags set to: 1
// MODE1: ZE ---> zeEventCreate(ZeEventPool, &ZeEventDesc, &ZeEvent)
// MODE1: ZE ---> zeCommandListAppendWaitOnEvents(CommandList->first, 1, &ZeEvent)
// MODE1-NEXT: ZE ---> zeCommandListAppendSignalEvent(CommandList->first, HostVisibleEvent->ZeEvent)
// MODE1: Completed all kernels

// With the SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 mode look for pattern that
// creates host-visible event just before command-list submission.
//
// MODE2: ze_event_pool_desc_t flags set to: 1
// MODE2: ZE ---> zeEventCreate(ZeEventPool, &ZeEventDesc, &ZeEvent)
// MODE2: ZE ---> zeCommandListAppendBarrier(CommandList->first, HostVisibleEvent->ZeEvent, 0, nullptr)
// MODE2: ZE ---> zeCommandListClose(CommandList->first)
// MODE2: ZE ---> zeCommandQueueExecuteCommandLists(ZeCommandQueue, 1, &ZeCommandList, CommandList->second.ZeFence)
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
