// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urEventWaitTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                         nullptr, &src_buffer));
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                         nullptr, &dst_buffer));
        input.assign(count, 42);
        ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, src_buffer, false, 0,
                                               size, input.data(), 0, nullptr,
                                               &event));
        ASSERT_SUCCESS(urEventWait(1, &event));
    }

    void TearDown() override {
        if (src_buffer) {
            EXPECT_SUCCESS(urMemRelease(src_buffer));
        }
        if (dst_buffer) {
            EXPECT_SUCCESS(urMemRelease(dst_buffer));
        }
        if (event) {
            EXPECT_SUCCESS(urEventRelease(event));
        }
        urQueueTest::TearDown();
    }

    const size_t count = 1024;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t src_buffer = nullptr;
    ur_mem_handle_t dst_buffer = nullptr;
    ur_event_handle_t event = nullptr;
    std::vector<uint32_t> input;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventWaitTest);

TEST_P(urEventWaitTest, Success) {
    ur_event_handle_t event1 = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, &event1));
    std::vector<uint32_t> output(count, 1);
    ur_event_handle_t event2 = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer, false, 0, size,
                                          output.data(), 0, nullptr, &event2));
    std::vector<ur_event_handle_t> events{event1, event2};
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(
        urEventWait(static_cast<uint32_t>(events.size()), events.data()));
    ASSERT_EQ(input, output);

    EXPECT_SUCCESS(urEventRelease(event1));
    EXPECT_SUCCESS(urEventRelease(event2));
}

using urEventWaitNegativeTest = uur::urQueueTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventWaitNegativeTest);

TEST_P(urEventWaitNegativeTest, ZeroSize) {
    ur_event_handle_t event = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE, urEventWait(0, &event));
}

TEST_P(urEventWaitNegativeTest, InvalidNullPointerEventList) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEventWait(1, nullptr));
}
