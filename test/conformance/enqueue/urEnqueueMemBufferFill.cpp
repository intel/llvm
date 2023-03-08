// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urEnqueueMemBufferFillTest = uur::urMemBufferQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferFillTest);

TEST_P(urEnqueueMemBufferFillTest, Success) {
    const uint32_t pattern = 0xdeadbeef;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, &pattern,
                                          sizeof(pattern), 0, size, 0, nullptr,
                                          nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    for (unsigned i = 0; i < count; ++i) {
        ASSERT_EQ(output[i], pattern) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferFillTest, SuccessPartialFill) {
    const std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
    const uint32_t pattern = 0xdeadbeef;
    const size_t partial_fill_size = size / 2;
    const size_t fill_count = count / 2;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, &pattern,
                                          sizeof(pattern), 0, partial_fill_size,
                                          0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    for (size_t i = 0; i < count - fill_count; ++i) {
        ASSERT_EQ(output[i], pattern) << "Result mismatch at index: " << i;
    }

    for (size_t i = fill_count; i < count; ++i) {
        ASSERT_EQ(output[i], 42) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferFillTest, SuccessOffset) {
    const std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
    const uint32_t pattern = 0xdeadbeef;
    const size_t offset_size = size / 2;
    const size_t offset_count = count / 2;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, &pattern,
                                          sizeof(pattern), offset_size,
                                          offset_size, 0, nullptr, nullptr));
    std::vector<uint32_t> output(count, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    for (size_t i = 0; i < offset_count; ++i) {
        ASSERT_EQ(output[i], 42) << "Result mismatch at index: " << i;
    }

    for (size_t i = offset_count; i < count; ++i) {
        ASSERT_EQ(output[i], pattern) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferFillTest, InvalidNullHandleQueue) {
    const uint32_t pattern = 0xdeadbeef;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferFill(nullptr, buffer, &pattern,
                                            sizeof(pattern), 0, size, 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferFillTest, InvalidNullHandleBuffer) {
    const uint32_t pattern = 0xdeadbeef;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferFill(queue, nullptr, &pattern,
                                            sizeof(pattern), 0, size, 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferFillTest, InvalidNullHandlePointerPattern) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueMemBufferFill(queue, buffer, nullptr,
                                            sizeof(uint32_t), 0, size, 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferFillTest, InvalidNullPtrEventWaitList) {
    const uint32_t pattern = 0xdeadbeef;
    ASSERT_EQ_RESULT(urEnqueueMemBufferFill(queue, buffer, &pattern,
                                            sizeof(uint32_t), 0, size, 1,
                                            nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemBufferFill(queue, buffer, &pattern,
                                            sizeof(uint32_t), 0, size, 0,
                                            &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}

TEST_P(urEnqueueMemBufferFillTest, InvalidSize) {
    const uint32_t pattern = 0xdeadbeef;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemBufferFill(queue, buffer, &pattern,
                                            sizeof(pattern), 1, size, 0,
                                            nullptr, nullptr));
}

using urEnqueueMemBufferFillMultiDeviceTest =
    uur::urMultiDeviceMemBufferQueueTest;

TEST_F(urEnqueueMemBufferFillMultiDeviceTest, FillReadDifferentQueues) {
    // First queue does a fill.
    const uint32_t input = 42;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &input,
                                          sizeof(input), 0, size, 0, nullptr,
                                          nullptr));

    // Wait for the queue to finish executing.
    EXPECT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));

    // Then the remaining queues do blocking reads from the buffer. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];
        std::vector<uint32_t> output(count, 0);
        ASSERT_SUCCESS(urEnqueueMemBufferRead(
            queue, buffer, true, 0, size, output.data(), 0, nullptr, nullptr));
        for (unsigned j = 0; j < count; ++j) {
            ASSERT_EQ(input, output[j])
                << "Result on queue " << i << " did not match at index " << j
                << "!";
        }
    }
}
