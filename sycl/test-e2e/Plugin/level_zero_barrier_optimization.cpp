// REQUIRES: level_zero
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// Test to check that we don't insert unnecessary L0 commands for
// queue::ext_oneapi_submit_barrier() when we have in-order queue.

#include <sycl/sycl.hpp>

class TestKernel;
sycl::event submitKernel(sycl::queue &Q) {
  return Q.submit(
      [&](sycl::handler &cgh) { cgh.single_task<TestKernel>([]() {}); });
}

void verifyEvent(sycl::event &E) {
  assert(E.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete);
}

int main() {
  sycl::queue Q1({sycl::property::queue::in_order{}});
  sycl::queue Q2({sycl::property::queue::in_order{}});
  sycl::queue Q3({sycl::property::queue::in_order{},
                  sycl::property::queue::enable_profiling{}});

  // Any dependencies from the same queue are filtered out on the SYCL runtime
  // level, only cases with cross-queue events need to be checked here.
  {
    // Test case 1 - events in the barrier's waitlist are from different queues.
    std::cout << "Test1" << std::endl;
    auto EventA = submitKernel(Q1);
    auto EventB = submitKernel(Q2);

    // CHECK: Test1
    // CHECK: ---> piEnqueueEventsWaitWithBarrier(
    // CHECK: ZE ---> zeEventCreate
    // CHECK: ZE ---> zeCommandListAppendWaitOnEvents
    // CHECK: ZE ---> zeCommandListAppendSignalEvent
    // CHECK: ) ---> 	pi_result : PI_SUCCESS
    auto BarrierEvent = Q2.ext_oneapi_submit_barrier({EventA, EventB});
    BarrierEvent.wait();

    verifyEvent(EventA);
    verifyEvent(EventB);
  }
  {
    // Test case 2 - events in the barrier's waitlist are from the same queue
    // Q2, but submission to the different queue Q1 which is synced.
    std::cout << "Test2" << std::endl;
    Q1.wait();
    auto EventA = submitKernel(Q2);
    auto EventB = submitKernel(Q2);

    // CHECK: Test2
    // CHECK: ---> piEnqueueEventsWaitWithBarrier(
    // CHECK: ZE ---> {{zeEventCreate|zeEventHostReset}}
    // CHECK: ZE ---> zeCommandListAppendWaitOnEvents
    // CHECK: ZE ---> zeCommandListAppendSignalEvent
    // CHECK: ) ---> 	pi_result : PI_SUCCESS
    auto BarrierEvent = Q1.ext_oneapi_submit_barrier({EventA, EventB});
    BarrierEvent.wait();

    verifyEvent(EventA);
    verifyEvent(EventB);
  }
  {
    // Test case 3 - submit barrier after queue sync with profiling enabled,
    // i.e. last event = nullptr.
    std::cout << "Test3" << std::endl;
    auto EventA = submitKernel(Q2);
    auto EventB = submitKernel(Q3);
    Q2.wait();
    Q3.wait();
    // CHECK: Test3
    // CHECK: ---> piEnqueueEventsWaitWithBarrier(
    // CHECK: ZE ---> zeEventCreate
    // CHECK-NOT: ZE ---> zeCommandListAppendWaitOnEvents
    // CHECK-NOT: ZE ---> zeCommandListAppendSignalEvent
    // CHECK: ZE ---> zeCommandListAppendBarrier
    // CHECK: ) ---> 	pi_result : PI_SUCCESS
    auto BarrierEvent = Q3.ext_oneapi_submit_barrier({EventA, EventB});
    BarrierEvent.wait();

    verifyEvent(EventA);
    verifyEvent(EventB);
  }
  return 0;
}
