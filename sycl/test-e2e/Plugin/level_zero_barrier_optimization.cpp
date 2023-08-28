// REQUIRES: level_zero
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 ZE_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// Test to check that we don't insert unnecessary L0 commands for
// queue::ext_oneapi_submit_barrier() when we have in-order queue.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q1({sycl::property::queue::in_order{}});
  sycl::queue Q2({sycl::property::queue::in_order{}});

  // Test case 1 - events in the barrier's waitlist are from different queues.
  auto Event1 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel1>([]() {}); });
  auto Event2 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel2>([]() {}); });

  // CHECK: ---> piEnqueueEventsWaitWithBarrier(
  // CHECK: ZE ---> zeEventCreate
  // CHECK: ZE ---> zeCommandListAppendWaitOnEvents
  // CHECK: ZE ---> zeCommandListAppendSignalEvent
  // CHECK: ) ---> 	pi_result : PI_SUCCESS
  auto BarrierEvent1 = Q1.ext_oneapi_submit_barrier({Event1, Event2});
  BarrierEvent1.wait();

  // Check that kernel events are completed after waiting for barrier event.
  assert(Event1.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);
  assert(Event2.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);

  // Test case 2 - events in the barrier's waitlist are from the same queue.
  auto Event3 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel3>([]() {}); });
  auto Event4 = Q1.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel4>([]() {}); });

  // CHECK: ---> piEnqueueEventsWaitWithBarrier(
  // CHECK-NOT: ZE ---> zeCommandListAppendWaitOnEvents
  // CHECK-NOT: ZE ---> zeCommandListAppendSignalEvent
  // CHECK-NOT: ZE ---> zeCommandListAppendBarrier
  // CHECK: ) ---> 	pi_result : PI_SUCCESS
  auto BarrierEvent2 = Q1.ext_oneapi_submit_barrier({Event3, Event4});
  BarrierEvent2.wait();

  // Check that kernel events are completed after waiting for barrier event.
  assert(Event3.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);
  assert(Event4.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);

  // Test case 3 - submit barrier after queue sync, i.e. last event = nullptr.
  auto Event5 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel5>([]() {}); });
  auto Event6 = Q2.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class kernel6>([]() {}); });
  Q2.wait();

  // CHECK: ---> piEnqueueEventsWaitWithBarrier(
  // CHECK: ZE ---> zeEventCreate
  // CHECK-NOT: ZE ---> zeCommandListAppendWaitOnEvents
  // CHECK: ZE ---> zeCommandListAppendSignalEvent
  // CHECK: ) ---> 	pi_result : PI_SUCCESS
  auto BarrierEvent3 = Q2.ext_oneapi_submit_barrier({Event5, Event6});
  BarrierEvent3.wait();

  // Check that kernel events are completed after waiting for barrier event.
  assert(Event5.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);
  assert(Event6.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);

  return 0;
}
