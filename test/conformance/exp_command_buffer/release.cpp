// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urCommandBufferReleaseExpTest =
    uur::command_buffer::urCommandBufferExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCommandBufferReleaseExpTest);

TEST_P(urCommandBufferReleaseExpTest, Success) {
    EXPECT_SUCCESS(urCommandBufferRetainExp(cmd_buf_handle));

    uint32_t prev_ref_count = 0;
    EXPECT_SUCCESS(
        uur::GetObjectReferenceCount(cmd_buf_handle, prev_ref_count));

    EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf_handle));

    uint32_t ref_count = 0;
    EXPECT_SUCCESS(uur::GetObjectReferenceCount(cmd_buf_handle, ref_count));

    EXPECT_GT(prev_ref_count, ref_count);
}

TEST_P(urCommandBufferReleaseExpTest, InvalidNullHandle) {
    EXPECT_EQ_RESULT(urCommandBufferReleaseExp(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

using urCommandBufferReleaseCommandExpTest =
    uur::command_buffer::urCommandBufferCommandExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCommandBufferReleaseCommandExpTest);

TEST_P(urCommandBufferReleaseCommandExpTest, Success) {
    EXPECT_SUCCESS(urCommandBufferRetainCommandExp(command_handle));

    uint32_t prev_ref_count = 0;
    EXPECT_SUCCESS(
        uur::GetObjectReferenceCount(command_handle, prev_ref_count));

    EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle));

    uint32_t ref_count = 0;
    EXPECT_SUCCESS(uur::GetObjectReferenceCount(command_handle, ref_count));

    EXPECT_GT(prev_ref_count, ref_count);
}

TEST_P(urCommandBufferReleaseCommandExpTest, ReleaseCmdBufBeforeHandle) {
    EXPECT_SUCCESS(urCommandBufferReleaseExp(updatable_cmd_buf_handle));

    // Ref count of `updatable_cmd_buf_handle` but shouldn't be destroyed
    // until all handles as destroyed.
    EXPECT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    EXPECT_SUCCESS(urQueueFinish(queue));
    updatable_cmd_buf_handle = nullptr;

    EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle));
    command_handle = nullptr;
}

TEST_P(urCommandBufferReleaseCommandExpTest, ReleaseCmdBufMultipleHandles) {
    EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle));
    command_handle = nullptr;

    // Ref count of `updatable_cmd_buf_handle` but should still be above zero
    EXPECT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    EXPECT_SUCCESS(urQueueFinish(queue));

    // Ref count of `updatable_cmd_buf_handle` but shouldn't be destroyed
    // until all handles as destroyed.
    EXPECT_SUCCESS(urCommandBufferReleaseExp(updatable_cmd_buf_handle));
    EXPECT_SUCCESS(urCommandBufferEnqueueExp(updatable_cmd_buf_handle, queue, 0,
                                             nullptr, nullptr));
    EXPECT_SUCCESS(urQueueFinish(queue));
    updatable_cmd_buf_handle = nullptr;

    EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle_2));
    command_handle_2 = nullptr;
}

TEST_P(urCommandBufferReleaseCommandExpTest, InvalidNullHandle) {
    EXPECT_EQ_RESULT(urCommandBufferReleaseCommandExp(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
