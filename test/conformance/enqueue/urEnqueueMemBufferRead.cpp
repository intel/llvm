// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urEnqueueMemBufferReadTest = uur::urMemBufferQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferReadTest);

TEST_P(urEnqueueMemBufferReadTest, Success) {
    std::vector<uint32_t> output(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferReadTest, InvalidNullHandleQueue) {
    std::vector<uint32_t> output(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferRead(nullptr, buffer, true, 0, size,
                                            output.data(), 0, nullptr,
                                            nullptr));
}

TEST_P(urEnqueueMemBufferReadTest, InvalidNullHandleBuffer) {
    std::vector<uint32_t> output(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferRead(queue, nullptr, true, 0, size,
                                            output.data(), 0, nullptr,
                                            nullptr));
}

TEST_P(urEnqueueMemBufferReadTest, InvalidNullPointerDst) {
    std::vector<uint32_t> output(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                            nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferReadTest, InvalidNullPtrEventWaitList) {
    std::vector<uint32_t> output(count, 42);
    ASSERT_EQ_RESULT(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                            output.data(), 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                            output.data(), 0, &validEvent,
                                            nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                            output.data(), 1, &inv_evt,
                                            nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}

TEST_P(urEnqueueMemBufferReadTest, InvalidSize) {
    std::vector<uint32_t> output(count, 42);
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemBufferRead(queue, buffer, true, 1, size,
                                            output.data(), 0, nullptr,
                                            nullptr));
}

TEST_P(urEnqueueMemBufferReadTest, Blocking) {
    constexpr const size_t memSize = 10u;
    constexpr const size_t bytes = memSize * sizeof(int);
    const int data[memSize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int output[memSize] = {};

    ur_mem_handle_t memObj;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, bytes,
                                     nullptr, &memObj));

    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, memObj, true, 0, bytes, data,
                                           0, nullptr, nullptr));

    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, memObj, true, 0, bytes, output,
                                          0, nullptr, nullptr));

    bool isSame =
        std::equal(std::begin(output), std::end(output), std::begin(data));
    EXPECT_TRUE(isSame);
    if (!isSame) {
        std::for_each(std::begin(output), std::end(output),
                      [](int &elem) { std::cout << elem << ","; });
        std::cout << std::endl;
    }
}

TEST_P(urEnqueueMemBufferReadTest, NonBlocking) {
    constexpr const size_t memSize = 10u;
    constexpr const size_t bytes = memSize * sizeof(int);
    const int data[memSize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int output[memSize] = {};

    ur_mem_handle_t memObj;
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, bytes,
                                     nullptr, &memObj));

    ur_event_handle_t cpIn, cpOut;
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, memObj, false, 0, bytes, data,
                                           0, nullptr, &cpIn));
    ASSERT_NE(cpIn, nullptr);

    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, memObj, false, 0, bytes,
                                          output, 0, nullptr, &cpOut));
    ASSERT_NE(cpOut, nullptr);

    ASSERT_SUCCESS(urEventWait(1, &cpOut));

    bool isSame =
        std::equal(std::begin(output), std::end(output), std::begin(data));
    EXPECT_TRUE(isSame);
    if (!isSame) {
        std::for_each(std::begin(output), std::end(output),
                      [](int &elem) { std::cout << elem << ","; });
        std::cout << std::endl;
    }
}

using urEnqueueMemBufferReadMultiDeviceTest =
    uur::urMultiDeviceMemBufferQueueTest;

TEST_F(urEnqueueMemBufferReadMultiDeviceTest, WriteReadDifferentQueues) {
    // First queue does a blocking write of 42 into the buffer.
    std::vector<uint32_t> input(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    // Then the remaining queues do blocking reads from the buffer. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];
        std::vector<uint32_t> output(count, 0);
        ASSERT_SUCCESS(urEnqueueMemBufferRead(
            queue, buffer, true, 0, size, output.data(), 0, nullptr, nullptr));
        ASSERT_EQ(input, output)
            << "Result on queue " << i << " did not match!";
    }
}
