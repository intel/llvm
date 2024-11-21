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
    const auto device_handle = std::get<0>(info.param).device;
    const auto platform_device_name =
        uur::GetPlatformAndDeviceName(device_handle);
    std::stringstream test_name;
    test_name << platform_device_name << "__size__"
              << std::get<1>(info.param).size << "__patternSize__"
              << std::get<1>(info.param).pattern_size;
    return test_name.str();
}

struct urEnqueueUSMFillTestWithParam
    : uur::urQueueTestWithParam<testParametersFill> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::SetUp());

        size = std::get<1>(GetParam()).size;
        host_mem = std::vector<uint8_t>(size);
        pattern_size = std::get<1>(GetParam()).pattern_size;
        pattern = std::vector<uint8_t>(pattern_size);
        uur::generateMemFillPattern(pattern);

        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
        if (!device_usm) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ASSERT_SUCCESS(
            urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &ptr));
    }

    void TearDown() override {
        if (ptr) {
            EXPECT_SUCCESS(urUSMFree(context, ptr));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::TearDown());
    }

    void verifyData() {
        ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, host_mem.data(), ptr,
                                          size, 0, nullptr, nullptr));

        size_t pattern_index = 0;
        for (size_t i = 0; i < size; ++i) {

            uint8_t *host_ptr = host_mem.data();
            ASSERT_TRUE((*(host_ptr + i) == pattern[pattern_index]));

            ++pattern_index;
            if (pattern_index % pattern.size() == 0) {
                pattern_index = 0;
            }
        }
    }

    size_t size;
    size_t pattern_size;
    std::vector<uint8_t> pattern;
    std::vector<uint8_t> host_mem;
    void *ptr{nullptr};
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

UUR_DEVICE_TEST_SUITE_P(urEnqueueUSMFillTestWithParam,
                        testing::ValuesIn(test_cases),
                        printFillTestString<urEnqueueUSMFillTestWithParam>);

TEST_P(urEnqueueUSMFillTestWithParam, Success) {

    ur_event_handle_t event = nullptr;

    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, pattern_size, pattern.data(),
                                    size, 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));

    ASSERT_SUCCESS(urEventWait(1, &event));
    ur_event_status_t event_status;
    ASSERT_SUCCESS(uur::GetEventInfo<ur_event_status_t>(
        event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS, event_status));
    ASSERT_EQ(event_status, UR_EVENT_STATUS_COMPLETE);
    EXPECT_SUCCESS(urEventRelease(event));

    ASSERT_NO_FATAL_FAILURE(verifyData());
}

struct urEnqueueUSMFillNegativeTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
        if (!device_usm) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ASSERT_SUCCESS(
            urUSMDeviceAlloc(context, device, nullptr, nullptr, size, &ptr));
    }

    void TearDown() override {
        if (ptr) {
            EXPECT_SUCCESS(urUSMFree(context, ptr));
        }

        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::TearDown());
    }

    static constexpr size_t size = 16;
    static constexpr size_t pattern_size = 4;
    std::vector<uint8_t> pattern{0x01, 0x02, 0x03, 0x04};
    void *ptr{nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueUSMFillNegativeTest);

TEST_P(urEnqueueUSMFillNegativeTest, InvalidNullQueueHandle) {
    ASSERT_EQ_RESULT(urEnqueueUSMFill(nullptr, ptr, pattern_size,
                                      pattern.data(), size, 0, nullptr,
                                      nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueUSMFillNegativeTest, InvalidNullPtr) {

    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, nullptr, pattern_size,
                                      pattern.data(), size, 0, nullptr,
                                      nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueUSMFillNegativeTest, InvalidSize) {
    /* size is 0 */
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, pattern_size, pattern.data(),
                                      0, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);

    /* size is not a multiple of pattern_size */
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, pattern_size, pattern.data(),
                                      7, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFillNegativeTest, OutOfBounds) {
    size_t out_of_bounds = size + 1;
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, pattern_size, pattern.data(),
                                      out_of_bounds, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFillNegativeTest, invalidPatternSize) {
    /* pattern_size is 0 */
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, 0, pattern.data(), size, 0,
                                      nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);

    /* pattern_size is not a power of 2 */
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, 3, pattern.data(), size, 0,
                                      nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);

    /* pattern_size is larger than size */
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, 32, pattern.data(), size, 0,
                                      nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFillNegativeTest, InvalidEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, pattern_size, pattern.data(),
                                      size, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, pattern_size, pattern.data(),
                                      size, 0, &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueUSMFill(queue, ptr, pattern_size, pattern.data(),
                                      size, 1, &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
    ASSERT_SUCCESS(urEventRelease(validEvent));
}
