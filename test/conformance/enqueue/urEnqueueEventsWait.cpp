// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

struct urEnqueueEventsWaitTest : uur::urMultiQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                         nullptr, &src_buffer));
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                         nullptr, &dst_buffer));
        input.assign(count, 42);
        ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue1, src_buffer, true, 0,
                                               size, input.data(), 0, nullptr,
                                               nullptr));
    }

    void TearDown() override {
        if (src_buffer) {
            EXPECT_SUCCESS(urMemRelease(src_buffer));
        }
        if (dst_buffer) {
            EXPECT_SUCCESS(urMemRelease(dst_buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueTest::TearDown());
    }

    const size_t count = 1024;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t src_buffer = nullptr;
    ur_mem_handle_t dst_buffer = nullptr;
    std::vector<uint32_t> input;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueEventsWaitTest);

TEST_P(urEnqueueEventsWaitTest, Success) {
    ur_event_handle_t event1 = nullptr;
    ur_event_handle_t waitEvent = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferCopy(queue1, src_buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, &event1));
    ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 1, &event1, &waitEvent));
    ASSERT_SUCCESS(urQueueFlush(queue1));
    ASSERT_SUCCESS(urQueueFlush(queue2));
    ASSERT_SUCCESS(urEventWait(1, &waitEvent));

    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue1, dst_buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    ASSERT_EQ(input, output);

    ur_event_handle_t event2 = nullptr;
    input.assign(count, 420);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue2, src_buffer, true, 0, size,
                                           input.data(), 0, nullptr, &event2));
    ASSERT_SUCCESS(urEventWait(1, &event2));
    ASSERT_SUCCESS(urEnqueueMemBufferCopy(queue2, src_buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue2, dst_buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    ASSERT_EQ(input, output);
    EXPECT_SUCCESS(urEventRelease(event1));
    EXPECT_SUCCESS(urEventRelease(waitEvent));
    EXPECT_SUCCESS(urEventRelease(event2));
}

TEST_P(urEnqueueEventsWaitTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueEventsWait(nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueEventsWaitTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueEventsWait(queue1, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue1, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueEventsWait(queue1, 0, &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueEventsWait(queue1, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}
