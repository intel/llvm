// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urQueueEndGraphCaptureExpTest = uur::urGraphPopulatedExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urQueueEndGraphCaptureExpTest);

TEST_P(urQueueEndGraphCaptureExpTest, SuccessSameGraph) {
  ur_exp_graph_handle_t sameGraph = nullptr;
  ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &sameGraph));
  ASSERT_EQ(graph, sameGraph);
}

TEST_P(urQueueEndGraphCaptureExpTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urQueueEndGraphCaptureExp(nullptr, &graph));
}

TEST_P(urQueueEndGraphCaptureExpTest, InvalidNullPtrGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urQueueEndGraphCaptureExp(queue, nullptr));
}
