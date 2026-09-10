// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ./timestamp_recording-test
// REQUIRES: v2

#include <set>

#include <ze_api.h>

#include "../ze_helpers.hpp"
#include "uur/fixtures.h"

struct urEnqueueTimestampRecordingExpTest : uur::urQueueTest {
  void SetUp() override {
    // Required when this test is linked statically with the Level Zero loader,
    // the driver would not be initialized otherwise.
    zeInit(ZE_INIT_FLAG_GPU_ONLY);
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueTimestampRecordingExpTest);

// An event whose timestamp write is still in flight must not be recycled, no
// matter when the application drops its reference to it.
TEST_P(urEnqueueTimestampRecordingExpTest, PendingWriteKeepsEventAlive) {
  auto zeGate = createZeEvent(context, device);
  ASSERT_NE(zeGate, nullptr);

  // A failed assertion returns from here, so signal the gate on every exit
  // path - the queue would otherwise wait for it forever while being released.
  struct signal_on_exit {
    ze_event_handle_t zeEvent;
    ~signal_on_exit() { zeEventHostSignal(zeEvent); }
  } gateGuard{zeGate.get()};

  ur_event_handle_t gate = nullptr;
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(
      reinterpret_cast<ur_native_handle_t>(zeGate.get()), context, nullptr,
      &gate));

  // Nothing enqueued below runs until the gate is signalled, so the writes stay
  // pending with no timing assumption.
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 1, &gate, nullptr));

  ur_event_handle_t first = nullptr;
  ASSERT_SUCCESS(
      urEnqueueTimestampRecordingExp(queue, false, 0, nullptr, &first));
  ASSERT_NE(first, nullptr);
  auto firstHandle = first;

  // Drop the application's reference while the write is still pending.
  ASSERT_SUCCESS(urEventRelease(first));
  first = nullptr;

  // The event must not have gone back to the pool, so the next recording has to
  // get a different one.
  ur_event_handle_t second = nullptr;
  ASSERT_SUCCESS(
      urEnqueueTimestampRecordingExp(queue, false, 0, nullptr, &second));
  ASSERT_NE(second, nullptr);
  ASSERT_NE(second, firstHandle);
  auto secondHandle = second;

  // Let the queue run; finishing it retires the events held for the writes.
  ASSERT_EQ(zeEventHostSignal(zeGate.get()), ZE_RESULT_SUCCESS);
  ASSERT_SUCCESS(urQueueFinish(queue));

  uint64_t endTime = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(second, UR_PROFILING_INFO_COMMAND_END,
                                         sizeof(endTime), &endTime, nullptr));
  ASSERT_NE(endTime, 0u);
  ASSERT_SUCCESS(urEventRelease(second));

  // The writes completed, so recycling resumes.
  ur_event_handle_t third = nullptr;
  ASSERT_SUCCESS(
      urEnqueueTimestampRecordingExp(queue, true, 0, nullptr, &third));
  ASSERT_TRUE(third == firstHandle || third == secondHandle)
      << "expected a recycled event";

  endTime = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(third, UR_PROFILING_INFO_COMMAND_END,
                                         sizeof(endTime), &endTime, nullptr));
  ASSERT_NE(endTime, 0u);

  ASSERT_SUCCESS(urEventRelease(third));
  ASSERT_SUCCESS(urEventRelease(gate));
}

// Events of completed writes have to be reclaimed without synchronizing the
// queue, otherwise they accumulate for as long as the queue lives.
TEST_P(urEnqueueTimestampRecordingExpTest, CompletedWritesAreReclaimed) {
  constexpr size_t iterations = 4;
  constexpr size_t recordingsPerIteration = 2;

  std::set<ur_event_handle_t> seen;
  for (size_t i = 0; i < iterations; i++) {
    ur_event_handle_t dropped = nullptr;
    ASSERT_SUCCESS(
        urEnqueueTimestampRecordingExp(queue, false, 0, nullptr, &dropped));
    seen.insert(dropped);
    ASSERT_SUCCESS(urEventRelease(dropped));

    ur_event_handle_t kept = nullptr;
    ASSERT_SUCCESS(
        urEnqueueTimestampRecordingExp(queue, false, 0, nullptr, &kept));
    seen.insert(kept);

    // Wait for the recording, not for the queue: the queue is in order, so both
    // writes of this iteration are done, but nothing retired the command list.
    ASSERT_SUCCESS(urEventWait(1, &kept));
    ASSERT_SUCCESS(urEventRelease(kept));
  }

  EXPECT_LT(seen.size(), iterations * recordingsPerIteration)
      << "no event was reused";

  // A recycled event must still report its own timestamp.
  ur_event_handle_t event = nullptr;
  ASSERT_SUCCESS(
      urEnqueueTimestampRecordingExp(queue, true, 0, nullptr, &event));

  uint64_t endTime = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_END,
                                         sizeof(endTime), &endTime, nullptr));
  ASSERT_NE(endTime, 0u);
  ASSERT_SUCCESS(urEventRelease(event));
}
