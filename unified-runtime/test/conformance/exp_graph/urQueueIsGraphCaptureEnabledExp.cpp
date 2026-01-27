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
