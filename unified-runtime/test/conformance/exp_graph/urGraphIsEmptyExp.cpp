// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
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
