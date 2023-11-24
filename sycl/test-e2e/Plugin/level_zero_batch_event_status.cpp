// See https://github.com/intel/llvm-test-suite/issues/906
// REQUIRES: gpu, level_zero

// RUN: %{build} -o %t.out

// Set batching to 4 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=4 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// level_zero_batch_test.cpp
//
// This tests the level zero plugin's kernel batching code.  It specifically
// tests that the current batch is submitted when an Event execution status
// request is made.  This test uses explicit SYCL_PI_LEVEL_ZERO_BATCH_SIZE=4
// to make sure that the batching is submitted when the piEventGetInfo is
// done, rather than some other dynamic batching criteria.
//
// CHECK: ---> piEnqueueKernelLaunch
// CHECK: ZE ---> zeCommandListAppendLaunchKernel
// Shouldn't have closed until we see a piEventGetInfo
// CHECK-NOT:  ZE ---> zeCommandListClose
// CHECK-NOT:  ZE ---> zeCommandQueueExecuteCommandLists
// CHECK: ---> piEventGetInfo
// Shouldn't see another piGetEventInfo until after closing command list
// CHECK-NOT: ---> piEventGetInfo
// Look for close and Execute after piEventGetInfo
// CHECK:  ZE ---> zeCommandListClose
// CHECK:  ZE ---> zeCommandQueueExecuteCommandLists
// CHECK: ---> piEventGetInfo
// CHECK-NOT: piEventsWait
// CHECK: ---> piEnqueueKernelLaunch
// CHECK: ZE ---> zeCommandListAppendLaunchKernel
// CHECK: ---> piQueueFinish
// Look for close and Execute after piQueueFinish
// CHECK:  ZE ---> zeCommandListClose
// CHECK:  ZE ---> zeCommandQueueExecuteCommandLists
// CHECK: ---> piEventGetInfo
// No close and execute here, should already have happened.
// CHECK-NOT:  ZE ---> zeCommandListClose
// CHECK-NOT:  ZE ---> zeCommandQueueExecuteCommandLists
// CHECK-NOT: Test Fail
// CHECK: Test Pass
// UNSUPPORTED: ze_debug

#include <cassert>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include <thread>

int main(void) {
  sycl::queue q{sycl::default_selector_v};
  std::vector<sycl::event> events(10);

  sycl::event ev1 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.single_task([=] {});
  });

  bool ev1_completed = false;
  int try_count = 0;
  while (true) {
    auto ev1_status =
        ev1.get_info<sycl::info::event::command_execution_status>();
    if (ev1_status == sycl::info::event_command_status::complete) {
      std::cout << "Ev1 has completed" << std::endl;
      ev1_completed = true;
      break;
    }

    std::cout << "Ev1 has not yet completed: ";
    switch (ev1_status) {
    case sycl::info::event_command_status::submitted:
      std::cout << "submitted";
      break;
    case sycl::info::event_command_status::running:
      std::cout << "running";
      break;
    default:
      std::cout << "unrecognized";
      break;
    }
    std::cout << std::endl;

    std::chrono::milliseconds timespan(300);
    std::this_thread::sleep_for(timespan);

    try_count += 1;
    if (try_count > 10) {
      ev1.wait();
    }
  }
  assert(ev1_completed);

  sycl::event ev2 = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.single_task([=] {});
  });
  q.wait();

  auto ev2_status = ev2.get_info<sycl::info::event::command_execution_status>();
  if (ev2_status != sycl::info::event_command_status::complete) {
    std::cout << "Test Fail" << std::endl;
    exit(1);
  }

  std::cout << "Ev2 has completed" << std::endl;
  std::cout << "Test Pass" << std::endl;
  return 0;
}
