// REQUIRES: gpu, level_zero

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER

// Checks that with L0 device-scope events enabled the only host-visible L0
// event created is at the end of all kernels submission, when host waits for
// the last kernel's event.
//
// CHECK-LABEL: Submitted all kernels
// CHECK: ---> piEventsWait(
// CHECK-NEXT:        <unknown> : 1
// CHECK: ZE ---> zeEventCreate(ZeEventPool, &ZeEventDesc, &ZeHostVisibleEvent)
// CHECK: ZE ---> zeCommandListAppendWaitOnEvents(CommandList->first, 1, &ZeEvent)
// CHECK-NEXT: ZE ---> zeCommandListAppendSignalEvent(CommandList->first,
// ZeHostVisibleEvent)
// CHECK: Completed all kernels

#include <CL/sycl.hpp>

int main(int argc, char **argv) {
  cl::sycl::gpu_selector device_selector;
  cl::sycl::queue queue(device_selector);

  int N = (argc >= 2 ? std::atoi(argv[1]) : 100);
  std::cout << N << " kernels" << std::endl;

  cl::sycl::event e; // completed event
  for (int i = 0; i < N; i++) {
    e = queue.submit([&](cl::sycl::handler &h) {
      h.depends_on(e);
      h.single_task<class kernel>([=] {});
    });
  } // for

  std::cout << "Submitted all kernels" << std::endl;
  e.wait(); // Waits for the last kernel to complete.
  std::cout << "Completed all kernels" << std::endl;
  return 0;
}