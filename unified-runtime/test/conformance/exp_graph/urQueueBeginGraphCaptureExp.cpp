// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urQueueBeginGraphCaptureExpTest = uur::urGraphExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueBeginGraphCaptureExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueBeginGraphCaptureExpTest, Success) {
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue));

  ur_exp_graph_handle_t graph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &graph));
  ASSERT_SUCCESS(urGraphDestroyExp(graph));
}

TEST_P(urQueueBeginGraphCaptureExpTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueBeginGraphCaptureExp(nullptr));
}

struct urQueueBeginGraphCaptureExpCapturingTest : uur::urGraphSupportedExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::SetUp());

    ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue));
  }

  void TearDown() override {
    // TearDown also runs when SetUp stopped before the queue was created.
    bool isCapturing = false;
    if (queue &&
        urQueueIsGraphCaptureEnabledExp(queue, &isCapturing) ==
            UR_RESULT_SUCCESS &&
        isCapturing) {
      EXPECT_SUCCESS(urQueueEndGraphCaptureExp(queue, &graph));
    }
    capturedEvent = nullptr;
    if (graph) {
      EXPECT_SUCCESS(urGraphDestroyExp(graph));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::TearDown());
  }

  uur::raii::Event capturedEvent = nullptr;
  ur_exp_graph_handle_t graph = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueBeginGraphCaptureExpCapturingTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueBeginGraphCaptureExpCapturingTest, InvalidQueueFinish) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphCaptureErrorsMinL0DriverVersion);

  // Request an event so that a command is actually recorded.
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, capturedEvent.ptr()));

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED,
                   urQueueFinish(queue));
}

TEST_P(urQueueBeginGraphCaptureExpCapturingTest, InvalidEventWait) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphEventWaitErrorsMinL0DriverVersion);

  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, capturedEvent.ptr()));

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED,
                   urEventWait(1, capturedEvent.ptr()));
}

TEST_P(urQueueBeginGraphCaptureExpCapturingTest, InvalidBlockingEnqueue) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphCaptureErrorsMinL0DriverVersion);

  ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
  if (!hostUSMSupport) {
    GTEST_SKIP() << "Host USM is not supported";
  }

  constexpr size_t size = 64;
  void *src = nullptr;
  void *dst = nullptr;
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, size, &src));
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, size, &dst));

  ur_result_t result =
      urEnqueueUSMMemcpy(queue, true, dst, src, size, 0, nullptr, nullptr);

  ASSERT_SUCCESS(urUSMFree(context, src));
  ASSERT_SUCCESS(urUSMFree(context, dst));
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED, result);
}

using urQueueBeginGraphCaptureExpMultiQueueTest =
    uur::urGraphMultiQueueCaptureExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueBeginGraphCaptureExpMultiQueueTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueBeginGraphCaptureExpMultiQueueTest, InvalidMergeAttempt) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphForkErrorsMinL0DriverVersion);

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue2));

  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue2, 0, nullptr, capturedEvent.ptr()));

  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT,
      urEnqueueEventsWait(queue1, 1, capturedEvent.ptr(), nullptr));
}

TEST_P(urQueueBeginGraphCaptureExpMultiQueueTest,
       InvalidInternalEventWaitAfterCapture) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphInternalEventErrorsMinL0DriverVersion);

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue1, 0, nullptr, capturedEvent.ptr()));
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph1));

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT,
                   urEventWait(1, capturedEvent.ptr()));
}

TEST_P(urQueueBeginGraphCaptureExpMultiQueueTest,
       InvalidInternalEventOnNonCapturingQueue) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphInternalEventErrorsMinL0DriverVersion);

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue1, 0, nullptr, capturedEvent.ptr()));
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph1));

  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT,
      urEnqueueEventsWait(queue2, 1, capturedEvent.ptr(), nullptr));
}
