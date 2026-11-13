// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urQueueEndGraphCaptureExpTest = uur::urGraphExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueEndGraphCaptureExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueEndGraphCaptureExpTest, SuccessSameGraph) {
  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));
  ur_exp_graph_handle_t sameGraph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &sameGraph));
  ASSERT_EQ(graph, sameGraph);
}

// Regression test for out-of-order queue bug where begin/end capture
// would use different command lists.
TEST_P(urQueueEndGraphCaptureExpTest, SuccessAfterQueueOperations) {
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));
  ur_exp_graph_handle_t capturedGraph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &capturedGraph));
  ASSERT_EQ(graph, capturedGraph);
}

// Regression test for out-of-order queue bug where a reusable event passed to
// urEnqueueEventsWaitWithBarrierExt during capture was replaced with a newly
// allocated event.
TEST_P(urQueueEndGraphCaptureExpTest, SuccessBarrierExtWithReusableEvent) {
  ur_exp_event_desc_t desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr, 0};
  uur::raii::Event reusable{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, reusable.ptr()));
  ur_event_handle_t original = reusable;

  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));

  ur_exp_enqueue_ext_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr, 0};
  ur_result_t result = urEnqueueEventsWaitWithBarrierExt(
      queue, &props, 0, nullptr, reusable.ptr());

  ur_exp_graph_handle_t capturedGraph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &capturedGraph));
  ASSERT_EQ(graph, capturedGraph);

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(result);
  ASSERT_EQ(original, reusable.get());
}

TEST_P(urQueueEndGraphCaptureExpTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueEndGraphCaptureExp(nullptr, &graph));
}

TEST_P(urQueueEndGraphCaptureExpTest, InvalidNullPtrGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urQueueEndGraphCaptureExp(queue, nullptr));
}

TEST_P(urQueueEndGraphCaptureExpTest, InvalidNotCapturing) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphCaptureErrorsMinL0DriverVersion);

  ur_exp_graph_handle_t capturedGraph = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING,
                   urQueueEndGraphCaptureExp(queue, &capturedGraph));
}

TEST_P(urQueueEndGraphCaptureExpTest, InvalidCaptureAlreadyEnded) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphCaptureErrorsMinL0DriverVersion);

  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));
  ur_exp_graph_handle_t capturedGraph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &capturedGraph));

  ur_exp_graph_handle_t secondGraph = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_COMMAND_LIST_NOT_CAPTURING,
                   urQueueEndGraphCaptureExp(queue, &secondGraph));
}

using urQueueEndGraphCaptureExpMultiQueueTest =
    uur::urGraphMultiQueueCaptureExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urQueueEndGraphCaptureExpMultiQueueTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urQueueEndGraphCaptureExpMultiQueueTest, InvalidUnjoinedForks) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphForkErrorsMinL0DriverVersion);

  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
  UUR_RETURN_ON_FATAL_FAILURE(forkWithoutJoin());

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS,
                   urQueueEndGraphCaptureExp(queue1, &graph1));
  // The invalid graph is still returned so that it can be destroyed.
  ASSERT_NE(graph1, nullptr);

  // The capture is ended on the queue it began on and on the forked queue.
  bool isCapturing = true;
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue1, &isCapturing));
  ASSERT_FALSE(isCapturing);
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isCapturing));
  ASSERT_FALSE(isCapturing);
}

TEST_P(urQueueEndGraphCaptureExpMultiQueueTest, InvalidUnjoinedForksIntoGraph) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphForkErrorsMinL0DriverVersion);

  UUR_RETURN_ON_FATAL_FAILURE(recordGraphWithUnjoinedForks());

  bool isCapturing = true;
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue1, &isCapturing));
  ASSERT_FALSE(isCapturing);
  ASSERT_SUCCESS(urQueueIsGraphCaptureEnabledExp(queue2, &isCapturing));
  ASSERT_FALSE(isCapturing);
}

TEST_P(urQueueEndGraphCaptureExpMultiQueueTest, SuccessAfterUnjoinedForks) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphForkErrorsMinL0DriverVersion);

  UUR_RETURN_ON_FATAL_FAILURE(recordGraphWithUnjoinedForks());

  // Both queues can start a new, independent capture afterwards.
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue1));
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue1, &graph2));
  ASSERT_NE(graph2, nullptr);
  ASSERT_NE(graph2, graph1);

  ur_exp_graph_handle_t queue2Graph = nullptr;
  ASSERT_SUCCESS(urQueueBeginGraphCaptureExp(queue2));
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue2, &queue2Graph));
  ASSERT_SUCCESS(urGraphDestroyExp(queue2Graph));
}
