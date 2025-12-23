// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urQueueBeginCaptureIntoGraphExpTest = uur::urGraphExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urQueueBeginCaptureIntoGraphExpTest);

TEST_P(urQueueBeginCaptureIntoGraphExpTest, Success) {
  ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));
}

TEST_P(urQueueBeginCaptureIntoGraphExpTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueBeginCaptureIntoGraphExp(nullptr, graph));
}

TEST_P(urQueueBeginCaptureIntoGraphExpTest, InvalidNullHandleGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueBeginCaptureIntoGraphExp(queue, nullptr));
}

using urQueueBeginCaptureIntoPopulatedGraphExpTest =
    uur::urGraphPopulatedExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urQueueBeginCaptureIntoPopulatedGraphExpTest);

TEST_P(urQueueBeginCaptureIntoPopulatedGraphExpTest, InvalidNonEmptyGraph) {
  ASSERT_EQ(urQueueBeginCaptureIntoGraphExp(queue, graph),
            UR_RESULT_ERROR_INVALID_ARGUMENT);
}
