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
  void TearDown() override {
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

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));

  uur::raii::Event forkEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue1, 0, nullptr, forkEvent.ptr()));

  uur::raii::Event joinEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(queue2, 1, forkEvent.ptr(),
                                                joinEvent.ptr()));
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_TRUE(isEnabled);

  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue1, 1, joinEvent.ptr(), nullptr));

  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph));

  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue1, &isEnabled));
  ASSERT_FALSE(isEnabled);

  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isEnabled));
  ASSERT_FALSE(isEnabled);
}
