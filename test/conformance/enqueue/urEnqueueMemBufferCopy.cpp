// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urEnqueueMemBufferCopyTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                         nullptr, &src_buffer));
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                         nullptr, &dst_buffer));
        input.assign(count, 42);
        ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, src_buffer, true, 0, size,
                                               input.data(), 0, nullptr,
                                               nullptr));
    }

    void TearDown() override {
        if (src_buffer) {
            EXPECT_SUCCESS(urMemRelease(src_buffer));
        }
        if (src_buffer) {
            EXPECT_SUCCESS(urMemRelease(dst_buffer));
        }
        urQueueTest::TearDown();
    }

    const size_t count = 1024;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t src_buffer = nullptr;
    ur_mem_handle_t dst_buffer = nullptr;
    std::vector<uint32_t> input;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferCopyTest);

TEST_P(urEnqueueMemBufferCopyTest, Success) {
    ASSERT_SUCCESS(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    ASSERT_EQ(input, output);
}

TEST_P(urEnqueueMemBufferCopyTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferCopy(nullptr, src_buffer, dst_buffer, 0,
                                            0, size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyTest, InvalidNullHandleBufferSrc) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferCopy(queue, nullptr, dst_buffer, 0, 0,
                                            size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyTest, InvalidNullHandleBufferDst) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferCopy(queue, src_buffer, nullptr, 0, 0,
                                            size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                            size, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                            size, 0, &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                            size, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemBufferCopyTest, InvalidSize) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 1, 0,
                                            size, 0, nullptr, nullptr));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 1,
                                            size, 0, nullptr, nullptr));
}

using urEnqueueMemBufferCopyMultiDeviceTest =
    uur::urMultiDeviceMemBufferQueueTest;

TEST_F(urEnqueueMemBufferCopyMultiDeviceTest, CopyReadDifferentQueues) {
    // First queue does a fill.
    const uint32_t input = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &input,
                                          sizeof(input), 0, size, 0, nullptr,
                                          nullptr));

    // Then a copy.
    ur_mem_handle_t dst_buffer = nullptr;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                     nullptr, &dst_buffer));
    EXPECT_SUCCESS(urEnqueueMemBufferCopy(queues[0], buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, nullptr));

    // Wait for the queue to finish executing.
    EXPECT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));

    // Then the remaining queues do blocking reads from the buffer. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];
        std::vector<uint32_t> output(count, 0);
        EXPECT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer, true, 0, size,
                                              output.data(), 0, nullptr,
                                              nullptr));
        for (unsigned j = 0; j < count; ++j) {
            EXPECT_EQ(input, output[j])
                << "Result on queue " << i << " did not match at index " << j
                << "!";
        }
    }

    EXPECT_SUCCESS(urMemRelease(dst_buffer));
}
