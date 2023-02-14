// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueCreateWithNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueCreateWithNativeHandleTest);

TEST_P(urQueueCreateWithNativeHandleTest, InvalidNullHandleNativeQueue) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueCreateWithNativeHandle(nullptr, context, &queue));
}
