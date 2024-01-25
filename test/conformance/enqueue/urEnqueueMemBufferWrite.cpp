// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urEnqueueMemBufferWriteTest = uur::urMemBufferQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferWriteTest);

TEST_P(urEnqueueMemBufferWriteTest, Success) {
    std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteTest, SuccessWriteRead) {
    std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    for (size_t index = 0; index < count; index++) {
        ASSERT_EQ(input[index], output[index]);
    }
}

TEST_P(urEnqueueMemBufferWriteTest, InvalidNullHandleQueue) {
    std::vector<uint32_t> input(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferWrite(nullptr, buffer, true, 0, size,
                                             input.data(), 0, nullptr,
                                             nullptr));
}

TEST_P(urEnqueueMemBufferWriteTest, InvalidNullHandleBuffer) {
    std::vector<uint32_t> input(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferWrite(queue, nullptr, true, 0, size,
                                             input.data(), 0, nullptr,
                                             nullptr));
}

TEST_P(urEnqueueMemBufferWriteTest, InvalidNullPointerSrc) {
    std::vector<uint32_t> input(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                             nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteTest, InvalidNullPtrEventWaitList) {
    std::vector<uint32_t> input(count, 42);
    ASSERT_EQ_RESULT(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                             input.data(), 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                             input.data(), 0, &validEvent,
                                             nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                             input.data(), 1, &inv_evt,
                                             nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemBufferWriteTest, InvalidSize) {
    std::vector<uint32_t> output(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemBufferWrite(queue, buffer, true, 1, size,
                                             output.data(), 0, nullptr,
                                             nullptr));
}
