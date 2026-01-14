// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urQueueBeginGraphCaptureExpTest = uur::urGraphExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urQueueBeginGraphCaptureExpTest);

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
