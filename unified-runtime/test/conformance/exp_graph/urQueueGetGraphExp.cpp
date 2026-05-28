// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct urQueueGetGraphExpTest : uur::urGraphSupportedExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::SetUp());
    // zeCommandListGetGraphExp was added after initial graph support and does not have its own check.
    std::tuple<size_t, size_t, size_t> minL0DriverVersion = {1, 15, 37561};
    SKIP_IF_DRIVER_TOO_OLD("Level-Zero", minL0DriverVersion, platform, device);
  }

  void TearDown() override {
    if (graph) {
      ASSERT_SUCCESS(urGraphDestroyExp(graph));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::TearDown());
  }

  ur_exp_graph_handle_t graph = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueGetGraphExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueGetGraphExpTest, SuccessExplicitRecordingQueue) {
  ASSERT_SUCCESS(urGraphCreateExp(context, &graph));

  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));

  ur_exp_graph_handle_t retrievedGraph = nullptr;
  ASSERT_SUCCESS(urQueueGetGraphExp(queue, &retrievedGraph));
  ASSERT_EQ(retrievedGraph, graph);

  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &graph));
}

TEST_P(urQueueGetGraphExpTest, SuccessImplicitRecordingQueue) {
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue));

  ur_exp_graph_handle_t retrievedGraph1 = nullptr;
  ASSERT_SUCCESS(urQueueGetGraphExp(queue, &retrievedGraph1));
  ASSERT_NE(retrievedGraph1, nullptr);

  // In implicit mode, the handle gets cached after the first getGraph
  // call so query a second time to exercise the alternative codepath.
  ur_exp_graph_handle_t retrievedGraph2 = nullptr;
  ASSERT_SUCCESS(urQueueGetGraphExp(queue, &retrievedGraph2));
  ASSERT_NE(retrievedGraph2, nullptr);

  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &graph));
  ASSERT_EQ(retrievedGraph1, graph);
  ASSERT_EQ(retrievedGraph2, graph);
}

TEST_P(urQueueGetGraphExpTest, InvalidNonRecordingQueue) {
  ur_exp_graph_handle_t retrievedGraph = nullptr;
  ASSERT_EQ(urQueueGetGraphExp(queue, &retrievedGraph),
            UR_RESULT_ERROR_INVALID_OPERATION);
  ASSERT_EQ(retrievedGraph, nullptr);
}

TEST_P(urQueueGetGraphExpTest, InvalidAfterEndCapture) {
  ASSERT_SUCCESS(urGraphCreateExp(context, &graph));

  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &graph));

  ur_exp_graph_handle_t retrievedGraph = nullptr;
  ASSERT_EQ(urQueueGetGraphExp(queue, &retrievedGraph),
            UR_RESULT_ERROR_INVALID_OPERATION);
  ASSERT_EQ(retrievedGraph, nullptr);
}

TEST_P(urQueueGetGraphExpTest, InvalidNullHandleQueue) {
  ur_exp_graph_handle_t retrievedGraph = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueGetGraphExp(nullptr, &retrievedGraph));
}

TEST_P(urQueueGetGraphExpTest, InvalidNullPtrGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urQueueGetGraphExp(queue, nullptr));
}

struct urQueueGetGraphExpMultiQueueTest
    : uur::urGraphSupportedExpMultiQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpMultiQueueTest::SetUp());
    // Fork-join was initially broken with zeCommandListGetGraph
    std::tuple<size_t, size_t, size_t> minL0DriverVersion = {1, 15, 38146};
    SKIP_IF_DRIVER_TOO_OLD("Level-Zero", minL0DriverVersion, platform, device);

    // Fork-join with out-of-order queue broken due to multi command list capture bug
    if (getQueueFlag() & UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
      UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{});
    }
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

  void TestForkJoinPattern(bool isExplicitCapture) {
    // Submit event to switch command list for out-of-order queue with L0v2 adapter.
    // This triggers a bug where fork-join recording is not recognized by an out-of-order queue.
    uur::raii::Event preEvent = nullptr;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 0, nullptr, preEvent.ptr()));
    ASSERT_SUCCESS(urEventWait(1, preEvent.ptr()));

    if (isExplicitCapture) {
      ASSERT_SUCCESS(urGraphCreateExp(context, &graph));
      ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue1, graph));
    } else {
      ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
    }

    uur::raii::Event forkEvent = nullptr;
    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrier(queue1, 0, nullptr, forkEvent.ptr()));

    uur::raii::Event joinEvent = nullptr;
    ASSERT_SUCCESS(
        urEnqueueEventsWait(queue2, 1, forkEvent.ptr(), joinEvent.ptr()));

    ur_exp_graph_handle_t queue2Graph = nullptr;
    ASSERT_SUCCESS(urQueueGetGraphExp(queue2, &queue2Graph));

    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrier(queue1, 1, joinEvent.ptr(), nullptr));
    ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph));

    ASSERT_EQ(queue2Graph, graph);

    ur_exp_graph_handle_t queue2AfterGraph = nullptr;
    ASSERT_EQ(urQueueGetGraphExp(queue2, &queue2AfterGraph),
              UR_RESULT_ERROR_INVALID_OPERATION);
    ASSERT_EQ(queue2AfterGraph, nullptr);
  }

  ur_exp_graph_handle_t graph = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueGetGraphExpMultiQueueTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueGetGraphExpMultiQueueTest, ForkJoinPatternExplicit) {
  TestForkJoinPattern(true /* isExplicitCapture */);
}

TEST_P(urQueueGetGraphExpMultiQueueTest, ForkJoinPatternImplicit) {
  TestForkJoinPattern(false /* isExplicitCapture */);
}
