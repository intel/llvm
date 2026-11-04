// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ZE_ENABLE_TRACING_LAYER=1 ./barrier_optimization-test
// REQUIRES: v2

// Checks which ze calls urEnqueueEventsWaitWithBarrierExt makes: a barrier
// with pending events becomes wait-on-events + signal, a barrier with nothing
// pending on an in-order queue becomes a plain zeCommandListAppendBarrier.

#include "../ze_tracer_common.hpp"
#include "uur/fixtures.h"

// These handles are reset when inserting the barrier. 
static int waitOnEventsCount = 0;
static int signalEventCount = 0;
static int barrierCount = 0;

static std::shared_ptr<_zel_tracer_handle_t> tracer = [] {
  zel_core_callbacks_t prologue_callbacks{};
  prologue_callbacks.CommandList.pfnAppendWaitOnEventsCb =
      [](ze_command_list_append_wait_on_events_params_t *, ze_result_t, void *,
         void **) { waitOnEventsCount++; };
  prologue_callbacks.CommandList.pfnAppendSignalEventCb =
      [](ze_command_list_append_signal_event_params_t *, ze_result_t, void *,
         void **) { signalEventCount++; };
  prologue_callbacks.CommandList.pfnAppendBarrierCb =
      [](ze_command_list_append_barrier_params_t *, ze_result_t, void *,
         void **) { barrierCount++; };
  return enableTracing(std::move(prologue_callbacks), {});
}();

struct urBarrierOptimizationTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "foo";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &q1));
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &q2));
    ur_queue_properties_t props = {UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr,
                                   UR_QUEUE_FLAG_PROFILING_ENABLE};
    ASSERT_SUCCESS(urQueueCreate(context, device, &props, &q3));
  }

  void TearDown() override {
    for (auto e : events) {
      EXPECT_SUCCESS(urEventRelease(e));
    }
    for (auto q : {q1, q2, q3}) {
      if (q) {
        EXPECT_SUCCESS(urQueueRelease(q));
      }
    }
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::TearDown());
  }

  ur_event_handle_t launch(ur_queue_handle_t q) {
    size_t offset = 0, size = 1;
    ur_event_handle_t e = nullptr;
    EXPECT_SUCCESS(urEnqueueKernelLaunchWithArgsExp(
        q, kernel, 1, &offset, &size, &size, GetNumArgs(), GetArgs(), nullptr,
        0, nullptr, &e));
    events.push_back(e);
    return e;
  }

  void barrier(ur_queue_handle_t q, std::vector<ur_event_handle_t> waitList) {
    waitOnEventsCount = 0;
    signalEventCount = 0;
    barrierCount = 0;

    ur_event_handle_t e = nullptr;
    ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrierExt(
        q, nullptr, static_cast<uint32_t>(waitList.size()), waitList.data(),
        &e));
    events.push_back(e);
    ASSERT_SUCCESS(urEventWait(static_cast<uint32_t>(events.size()),
                               events.data()));
    for (auto ev : events) {
      ur_event_status_t status;
      ASSERT_SUCCESS(urEventGetInfo(ev, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                    sizeof(status), &status, nullptr));
      ASSERT_EQ(status, UR_EVENT_STATUS_COMPLETE);
    }
  }

  ur_queue_handle_t q1 = nullptr, q2 = nullptr, q3 = nullptr;
  std::vector<ur_event_handle_t> events;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urBarrierOptimizationTest);

// Barrier on Q2 waiting on pending events from Q1 and Q2: expect the barrier
// to be lowered to wait-on-events + a single signal, not a full barrier.
TEST_P(urBarrierOptimizationTest, EventsFromDifferentQueues) {
  auto a = launch(q1);
  auto b = launch(q2);
  UUR_RETURN_ON_FATAL_FAILURE(barrier(q2, {a, b}));
  EXPECT_GE(waitOnEventsCount, 1);
  EXPECT_EQ(signalEventCount, 1);
}

// Barrier on idle Q1 waiting only on pending events from Q2: expect
// wait-on-events + a single signal, not a full barrier.
TEST_P(urBarrierOptimizationTest, EventsFromOtherQueueOnIdleQueue) {
  ASSERT_SUCCESS(urQueueFinish(q1));
  auto a = launch(q2);
  auto b = launch(q2);
  UUR_RETURN_ON_FATAL_FAILURE(barrier(q1, {a, b}));
  EXPECT_GE(waitOnEventsCount, 1);
  EXPECT_EQ(signalEventCount, 1);
}

// Barrier with an empty wait list on profiling Q3 after all work finished:
// expect one plain zeCommandListAppendBarrier and no wait/signal calls.
TEST_P(urBarrierOptimizationTest, NoPendingWork) {
  launch(q2);
  launch(q3);
  ASSERT_SUCCESS(urQueueFinish(q2));
  ASSERT_SUCCESS(urQueueFinish(q3));
  UUR_RETURN_ON_FATAL_FAILURE(barrier(q3, {}));
  EXPECT_EQ(waitOnEventsCount, 0);
  EXPECT_EQ(signalEventCount, 0);
  EXPECT_EQ(barrierCount, 1);
}
