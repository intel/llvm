// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

struct testParametersFill {
    size_t size;
    size_t pattern_size;
};

template <typename T>
inline std::string
printFillTestString(const testing::TestParamInfo<typename T::ParamType> &info) {
    const auto device_handle = std::get<0>(info.param);
    const auto platform_device_name =
        uur::GetPlatformAndDeviceName(device_handle);
    std::stringstream test_name;
    test_name << platform_device_name << "__size__"
              << std::get<1>(info.param).size << "__patternSize__"
              << std::get<1>(info.param).pattern_size;
    return test_name.str();
}

struct urEnqueueMemBufferFillTest
    : uur::urQueueTestWithParam<testParametersFill> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urQueueTestWithParam<testParametersFill>::SetUp());
        size = std::get<1>(GetParam()).size;
        pattern_size = std::get<1>(GetParam()).pattern_size;
        pattern = std::vector<uint8_t>(pattern_size);
        uur::generateMemFillPattern(pattern);
        ASSERT_SUCCESS(urMemBufferCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                         size, nullptr, &buffer));
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(
            urQueueTestWithParam<testParametersFill>::TearDown());
    }

    void verifyData(std::vector<uint8_t> &output, size_t verify_size) {
        size_t pattern_index = 0;
        for (size_t i = 0; i < verify_size; ++i) {
            ASSERT_EQ(output[i], pattern[pattern_index])
                << "Result mismatch at index: " << i;

            ++pattern_index;
            if (pattern_index % pattern_size == 0) {
                pattern_index = 0;
            }
        }
    }

    ur_mem_handle_t buffer = nullptr;
    std::vector<uint8_t> pattern;
    size_t size;
    size_t pattern_size;
};

static std::vector<testParametersFill> test_cases{
    /* Everything set to 1 */
    {1, 1},
    /* pattern_size == size */
    {256, 256},
    /* pattern_size < size */
    {1024, 256},
    /* pattern sizes corresponding to some common scalar and vector types */
    {256, 4},
    {256, 8},
    {256, 16},
    {256, 32}};

UUR_TEST_SUITE_P(urEnqueueMemBufferFillTest, testing::ValuesIn(test_cases),
                 printFillTestString<urEnqueueMemBufferFillTest>);

TEST_P(urEnqueueMemBufferFillTest, Success) {
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, pattern.data(),
                                          pattern_size, 0, size, 0, nullptr,
                                          nullptr));
    std::vector<uint8_t> output(size, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    verifyData(output, size);
}
TEST_P(urEnqueueMemBufferFillTest, SuccessPartialFill) {
    if (size == 1) {
        // Can't partially fill one byte
        GTEST_SKIP();
    }
    const std::vector<uint8_t> input(size, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
    const size_t partial_fill_size = size / 2;
    // Make sure we don't end up with pattern_size > size
    pattern_size = pattern_size / 2;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, pattern.data(),
                                          pattern_size, 0, partial_fill_size, 0,
                                          nullptr, nullptr));
    std::vector<uint8_t> output(size, 1);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    // Check the first half matches the pattern and the second half remains untouched.
    verifyData(output, partial_fill_size);

    for (size_t i = partial_fill_size; i < size; ++i) {
        ASSERT_EQ(output[i], input[i]) << "Result mismatch at index: " << i;
    }
}

TEST_P(urEnqueueMemBufferFillTest, SuccessOffset) {
    if (size == 1) {
        // No room for an offset
        GTEST_SKIP();
    }
    const std::vector<uint8_t> input(size, 0);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));

    const size_t offset_size = size / 2;
    // Make sure we don't end up with pattern_size > size
    pattern_size = pattern_size / 2;
    ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, buffer, pattern.data(),
                                          pattern_size, offset_size,
                                          offset_size, 0, nullptr, nullptr));

    // Check the second half matches the pattern and the first half remains untouched.
    std::vector<uint8_t> output(offset_size);
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, offset_size,
                                          offset_size, output.data(), 0,
                                          nullptr, nullptr));
    verifyData(output, offset_size);

    ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, offset_size,
                                          output.data(), 0, nullptr, nullptr));
    for (size_t i = 0; i < offset_size; ++i) {
        ASSERT_EQ(output[i], input[i]) << "Result mismatch at index: " << i;
    }
}

using urEnqueueMemBufferFillNegativeTest = uur::urMemBufferQueueTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemBufferFillNegativeTest);

TEST_P(urEnqueueMemBufferFillNegativeTest, InvalidNullHandleQueue) {
    const uint32_t pattern = 0xdeadbeef;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferFill(nullptr, buffer, &pattern,
                                            sizeof(pattern), 0, size, 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferFillNegativeTest, InvalidNullHandleBuffer) {
    const uint32_t pattern = 0xdeadbeef;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemBufferFill(queue, nullptr, &pattern,
                                            sizeof(pattern), 0, size, 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferFillNegativeTest, InvalidNullHandlePointerPattern) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueMemBufferFill(queue, buffer, nullptr,
                                            sizeof(uint32_t), 0, size, 0,
                                            nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferFillNegativeTest, InvalidNullPtrEventWaitList) {
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

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemBufferFill(queue, buffer, &pattern,
                                            sizeof(uint32_t), 0, size, 1,
                                            &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemBufferFillNegativeTest, InvalidSize) {
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
