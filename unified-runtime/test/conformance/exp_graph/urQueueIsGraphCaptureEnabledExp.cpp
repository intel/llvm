// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct urQueueIsGraphCaptureEnabledExpTest : uur::urGraphSupportedExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::SetUp());

    ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue));
    isCapturing = true;
  }

  void TearDown() override {
    endGraphCapture();
    if (graph) {
      ASSERT_SUCCESS(urGraphDestroyExp(graph));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::TearDown());
  }

  void endGraphCapture() {
    if (isCapturing) {
      ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &graph));
      isCapturing = false;
    }
  }

  bool isCapturing = false;
  ur_exp_graph_handle_t graph = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueIsGraphCaptureEnabledExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueIsGraphCaptureEnabledExpTest, SuccessEnabled) {
  bool isEnabled = false;
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue, &isEnabled));
  ASSERT_TRUE(isEnabled);
}

TEST_P(urQueueIsGraphCaptureEnabledExpTest, SuccessDisabled) {
  endGraphCapture();

  bool isEnabled = false;
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue, &isEnabled));
  ASSERT_FALSE(isEnabled);
}

TEST_P(urQueueIsGraphCaptureEnabledExpTest, InvalidNullHandleQueue) {
  bool isEnabled = false;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueIsGraphCaptureEnabledExp(nullptr, &isEnabled));
}

TEST_P(urQueueIsGraphCaptureEnabledExpTest, InvalidNullPtrResult) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urQueueIsGraphCaptureEnabledExp(queue, nullptr));
}

struct urQueueIsGraphCaptureEnabledExpMultiQueueTest
    : uur::urGraphSupportedExpMultiQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpMultiQueueTest::SetUp());
  }
  void TearDown() override {
    bool isCaptureEnabled = false;
    if (queue1) {
      if (urQueueIsGraphCaptureEnabledExp(queue1, &isCaptureEnabled) ==
              UR_RESULT_SUCCESS &&
          isCaptureEnabled) {
        urQueueEndGraphCaptureExp(queue1, &graph);
      }
      urQueueFinish(queue1);
    }
    if (queue2) {
      urQueueFinish(queue2);
    }
    if (graph) {
      EXPECT_SUCCESS(urGraphDestroyExp(graph));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpMultiQueueTest::TearDown());
  }

  ur_exp_graph_handle_t graph = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueIsGraphCaptureEnabledExpMultiQueueTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

// Tests the fork-join pattern: When an operation is submitted to queue2 which
// is not recording with a dependent event from queue1 that is recording,
// queue2 temporarily transitions to a recording state (fork) which then must
// be joined back to queue1.
TEST_P(urQueueIsGraphCaptureEnabledExpMultiQueueTest, ForkJoinPattern) {
  bool isEnabled = false;

  // Submit event to switch command list for out-of-order queue with L0v2 adapter.
  // This triggers a bug where fork-join recording is not recognized by out-of-order queue.
  uur::raii::Event preEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 0, nullptr, preEvent.ptr()));
  ASSERT_SUCCESS(urEventWait(1, preEvent.ptr()));

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));

  uur::raii::Event forkEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue1, 0, nullptr, forkEvent.ptr()));

  uur::raii::Event joinEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueEventsWait(queue2, 1, forkEvent.ptr(), joinEvent.ptr()));
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_TRUE(isEnabled);

  ASSERT_SUCCESS(urEnqueueEventsWait(queue1, 1, joinEvent.ptr(), nullptr));

  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph));

  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue1, &isEnabled));
  ASSERT_FALSE(isEnabled);

  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_FALSE(isEnabled);
}

// Tests that, once an out-of-order queue has joined a capture via the fork-join
// pattern, operations subsequently submitted to it *without* an explicit
// dependency on the recording queue are still recorded. The secondary queue
// stays in the temporary recording state until the primary queue ends the
// capture. With the L0v2 out-of-order queue this would otherwise round-robin
// the dependency-less operation onto a non-capture command list, escaping the
// capture.
TEST_P(urQueueIsGraphCaptureEnabledExpMultiQueueTest,
       ForkJoinSubsequentOpsWithoutDependency) {
  bool isEnabled = false;

  // Advance the out-of-order queue's command list selection so the next
  // operation would not land on the dedicated capture command list by chance.
  uur::raii::Event preEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 0, nullptr, preEvent.ptr()));
  ASSERT_SUCCESS(urEventWait(1, preEvent.ptr()));

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));

  // Fork: queue1 produces an event that queue2 waits on, pulling queue2 into
  // the capture.
  uur::raii::Event forkEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue1, 0, nullptr, forkEvent.ptr()));

  uur::raii::Event joinEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueEventsWait(queue2, 1, forkEvent.ptr(), joinEvent.ptr()));
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_TRUE(isEnabled);

  // Subsequent operation on queue2 without any dependency on queue1. It must
  // remain part of the capture: queue2 should still report recording enabled.
  uur::raii::Event noDepEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 0, nullptr, noDepEvent.ptr()));
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_TRUE(isEnabled);

  // Join both queue2 operations back to queue1 and finish recording.
  ur_event_handle_t joinEvents[] = {joinEvent.get(), noDepEvent.get()};
  ASSERT_SUCCESS(urEnqueueEventsWait(queue1, 2, joinEvents, nullptr));

  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph));

  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue1, &isEnabled));
  ASSERT_FALSE(isEnabled);

  // The primary queue stopped recording, so queue2 must leave the temporary
  // recording state as well.
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_FALSE(isEnabled);
}
