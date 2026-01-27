// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
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
    UUR_RETURN_ON_FATAL_FAILURE(
        urGraphSupportedExpMultiQueueTest::SetUp());

    // Start capturing on queue1
    ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
    isCapturing = true;
  }

  void TearDown() override {
    if (isCapturing) {
      ur_exp_graph_handle_t graph = nullptr;
      EXPECT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph));
      if (graph) {
        EXPECT_SUCCESS(urGraphDestroyExp(graph));
      }
    }

    UUR_RETURN_ON_FATAL_FAILURE(
        urGraphSupportedExpMultiQueueTest::TearDown());
  }

  bool isCapturing = false;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueIsGraphCaptureEnabledExpMultiQueueTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

// Tests the fork-join pattern: When an operation is submitted to queue2 which
// is not recording with a dependent event from queue1 that is recording,
// queue2 temporarily transitions to a recording state (fork) until an event
// synchronization back to queue1 is performed (join).
TEST_P(urQueueIsGraphCaptureEnabledExpMultiQueueTest, ForkJoinPattern) {
  bool isEnabled = false;
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue1, &isEnabled));
  ASSERT_TRUE(isEnabled);

  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_FALSE(isEnabled);

  uur::raii::Event event1 = nullptr;
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue1, 0, nullptr, event1.ptr()));

  // Queue 2 submits a barrier with dependency on queue 1's event
  // This causes queue 2 to temporarily enter recording state (the "fork")
  uur::raii::Event event2 = nullptr;
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue2, 1, event1.ptr(), event2.ptr()));
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_TRUE(isEnabled);

  // Queue 1 submits another barrier with dependency on queue 2's event (join)
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue1, 1, event2.ptr(), nullptr));
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_FALSE(isEnabled);

  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue1, &isEnabled));
  ASSERT_TRUE(isEnabled);
}
