// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urQueueFinishTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueFinishTest);

TEST_P(urQueueFinishTest, Success) {
    constexpr size_t buffer_size = 1024;
    ur_mem_handle_t buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                     buffer_size, nullptr, &buffer));

    ur_event_handle_t event = nullptr;
    std::vector<uint8_t> data(buffer_size, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, /* blocking */ false,
                                           0, 1024, data.data(), 0, nullptr,
                                           &event));

    ASSERT_SUCCESS(urQueueFinish(queue));

    // check that enqueued commands have completed
    ur_event_status_t exec_status;
    ASSERT_SUCCESS(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(exec_status), &exec_status, nullptr));
    ASSERT_EQ(exec_status, UR_EXECUTION_INFO_COMPLETE);

    ASSERT_SUCCESS(urMemRelease(buffer));
    ASSERT_SUCCESS(urEventRelease(event));
}

TEST_P(urQueueFinishTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueFinish(nullptr));
}
