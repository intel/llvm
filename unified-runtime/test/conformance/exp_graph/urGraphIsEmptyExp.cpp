// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urGraphNonEmptyExpTest = uur::urGraphPopulatedExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphNonEmptyExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphNonEmptyExpTest, Success) {
  bool isEmpty = false;
  ASSERT_SUCCESS(urGraphIsEmptyExp(graph, &isEmpty));
  ASSERT_FALSE(isEmpty);
}

using urGraphEmptyExpTest = uur::urGraphExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphEmptyExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphEmptyExpTest, Success) {
  bool isEmpty = false;
  ASSERT_SUCCESS(urGraphIsEmptyExp(graph, &isEmpty));
  ASSERT_TRUE(isEmpty);
}

TEST_P(urGraphEmptyExpTest, InvalidNullHandleQueue) {
  bool isEmpty = false;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphIsEmptyExp(nullptr, &isEmpty));
}

TEST_P(urGraphEmptyExpTest, InvalidNullPtrResult) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphIsEmptyExp(graph, nullptr));
}

using urGraphIsEmptyExpMultiQueueTest = uur::urGraphMultiQueueCaptureExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphIsEmptyExpMultiQueueTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphIsEmptyExpMultiQueueTest, InvalidGraphUnjoinedForks) {
  UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(
      uur::graphForkErrorsMinL0DriverVersion);

  UUR_RETURN_ON_FATAL_FAILURE(recordGraphWithUnjoinedForks());

  bool isEmpty = false;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_GRAPH,
                   urGraphIsEmptyExp(graph1, &isEmpty));
}
