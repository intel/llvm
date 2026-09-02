// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Conformance tests for the descriptive graph capture error codes that were
// added to mirror the underlying Level-Zero graph extension error codes:
//   * UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED
//   * UR_RESULT_ERROR_GRAPH_CAPTURE_INVALIDATED
//   * UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT
//   * UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING
//   * UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS
//
// UR_RESULT_ERROR_INVALID_GRAPH and UR_RESULT_ERROR_GRAPH_INTERNAL_EVENT are
// intentionally not covered here: every public entry point that could
// surface them only does so for a graph/event handle that has already become
// invalid (e.g. a captured graph whose session failed, or a graph-internal
// counter-based event used after its graph went out of scope), and there is
// no way to reach that state through the public API without relying on
// undefined behaviour (use of a handle after its object was destroyed).

#include "fixtures.h"
#include "uur/raii.h"

using urQueueEndGraphCaptureNotCapturingExpTest = uur::urGraphSupportedExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueEndGraphCaptureNotCapturingExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

// Ending a graph capture on a queue that never began one must fail rather
// than silently succeed with an empty/invalid graph.
TEST_P(urQueueEndGraphCaptureNotCapturingExpTest, InvalidNotCapturing) {
  ur_exp_graph_handle_t graph = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING,
                   urQueueEndGraphCaptureExp(queue, &graph));
  ASSERT_EQ(graph, nullptr);
}

struct urGraphCaptureErrorsExpTest : uur::urGraphSupportedExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::SetUp());

    ur_device_usm_access_capability_flags_t deviceUSMSupport = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSMSupport));
    if (!deviceUSMSupport) {
      GTEST_SKIP() << "Device USM is not supported";
    }

    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &mem));
  }

  void TearDown() override {
    // If the capture session is still active (e.g. because it was
    // invalidated), close it before releasing the queue so the driver isn't
    // left with a dangling capturing command list.
    if (isCapturing) {
      ur_exp_graph_handle_t recordedGraph = nullptr;
      urQueueEndGraphCaptureExp(queue, &recordedGraph);
      if (recordedGraph) {
        urGraphDestroyExp(recordedGraph);
      }
    }

    if (mem) {
      ASSERT_SUCCESS(urUSMFree(context, mem));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::TearDown());
  }

  const size_t size = 64;
  void *mem = nullptr;
  bool isCapturing = false;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphCaptureErrorsExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

// A blocking command host-synchronizes the underlying command list. That is
// not a supported operation while the list is being captured into a graph,
// since capture only records commands rather than executing them.
TEST_P(urGraphCaptureErrorsExpTest, InvalidUnsupportedBlockingOperation) {
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue));
  isCapturing = true;

  const uint8_t pattern = 0x2a;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED,
      urEnqueueUSMFill(queue, mem, sizeof(pattern), &pattern, size, 0, nullptr,
                       nullptr));
}

// Once an operation invalidates a capture session, further operations on the
// same queue must keep failing until the capture is ended, rather than
// silently recording (or executing) commands into a corrupted graph.
TEST_P(urGraphCaptureErrorsExpTest, InvalidCaptureInvalidatedAfterFailure) {
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue));
  isCapturing = true;

  const uint8_t pattern = 0x2a;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_GRAPH_CAPTURE_UNSUPPORTED,
      urEnqueueUSMFill(queue, mem, sizeof(pattern), &pattern, size, 0, nullptr,
                       nullptr));

  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_GRAPH_CAPTURE_INVALIDATED,
      urEnqueueUSMFill(queue, mem, sizeof(pattern), &pattern, size, 0, nullptr,
                       nullptr));

  ur_exp_graph_handle_t recordedGraph = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_CAPTURE_INVALIDATED,
                   urQueueEndGraphCaptureExp(queue, &recordedGraph));
  ASSERT_EQ(recordedGraph, nullptr);
  isCapturing = false;
}

struct urGraphCaptureMultiQueueErrorsExpTest
    : uur::urGraphSupportedExpMultiQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        urGraphSupportedExpMultiQueueTest::SetUp());
  }

  void TearDown() override {
    if (queue1Capturing) {
      urQueueEndGraphCaptureExp(queue1, &graph1);
    }
    if (queue2Capturing) {
      urQueueEndGraphCaptureExp(queue2, &graph2);
    }
    if (graph1) {
      EXPECT_SUCCESS(urGraphDestroyExp(graph1));
    }
    if (graph2) {
      EXPECT_SUCCESS(urGraphDestroyExp(graph2));
    }
    UUR_RETURN_ON_FATAL_FAILURE(
        urGraphSupportedExpMultiQueueTest::TearDown());
  }

  ur_exp_graph_handle_t graph1 = nullptr;
  ur_exp_graph_handle_t graph2 = nullptr;
  bool queue1Capturing = false;
  bool queue2Capturing = false;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphCaptureMultiQueueErrorsExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

// A fork of the capture onto a secondary queue that is never joined back
// leaves the recorded graph structurally invalid (a dangling branch), so
// ending the capture must fail rather than silently returning an incomplete
// graph.
TEST_P(urGraphCaptureMultiQueueErrorsExpTest, InvalidUnjoinedForks) {
  // Ensure both queues have their command lists selected before recording,
  // matching other fork/join tests in this suite.
  uur::raii::Event preEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 0, nullptr, preEvent.ptr()));
  ASSERT_SUCCESS(urEventWait(1, preEvent.ptr()));

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
  queue1Capturing = true;

  // Fork: queue2 is pulled into the capture by waiting on an event produced
  // by queue1, but the fork is intentionally never joined back to queue1.
  uur::raii::Event forkEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue1, 0, nullptr, forkEvent.ptr()));
  ASSERT_SUCCESS(
      urEnqueueEventsWait(queue2, 1, forkEvent.ptr(), nullptr));

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS,
                   urQueueEndGraphCaptureExp(queue1, &graph1));
  ASSERT_EQ(graph1, nullptr);
  queue1Capturing = false;
}

// Waiting, from within one capture session, on an event that belongs to a
// different and still-active capture session would require merging the two
// independently recorded graphs, which is not supported.
TEST_P(urGraphCaptureMultiQueueErrorsExpTest, InvalidCaptureMergeAttempt) {
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
  queue1Capturing = true;
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue2));
  queue2Capturing = true;

  uur::raii::Event queue2Event = nullptr;
  ASSERT_SUCCESS(
      urEnqueueEventsWaitWithBarrier(queue2, 0, nullptr, queue2Event.ptr()));

  // queue2Event belongs to queue2's independent capture session. Waiting on
  // it from queue1's capture session attempts to merge the two sessions.
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_GRAPH_CAPTURE_MERGE_ATTEMPT,
      urEnqueueEventsWait(queue1, 1, queue2Event.ptr(), nullptr));
}
