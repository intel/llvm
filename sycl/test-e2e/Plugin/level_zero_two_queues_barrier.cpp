// REQUIRES: level_zero
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 ZE_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// Test to check that we don't insert unnecessary L0 commands for
// queue::ext_oneapi_submit_barrier() when synchronize over two in-order queues.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q1({sycl::property::queue::in_order{}});
  sycl::queue Q2({sycl::property::queue::in_order{}});

  auto Event1 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel1>([]() {}); });
  auto Event2 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel2>([]() {}); });
  auto BarrierEvent1 = Q1.ext_oneapi_submit_barrier(
      {Event2}); // Optimization will return Event2 as BarrierEvent1 here.
                 // Check that barriers are no-op.
                 // CHECK-NOT: ZE ---> zeCommandListAppendWaitOnEvents
                 // CHECK-NOT: ZE ---> zeCommandListAppendSignalEvent
                 // CHECK-NOT: ZE ---> zeCommandListAppendBarrier
  auto BarrierEvent2 = Q2.ext_oneapi_submit_barrier({BarrierEvent1});
  auto Event3 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel3>([]() {}); });
  Event3.wait();
  // Check that dependency chain was respected and events are completed.
  assert(Event1.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);
  assert(Event2.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);

  return 0;
}
