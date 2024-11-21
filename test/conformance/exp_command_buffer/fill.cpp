// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct testParametersFill {
    size_t size;
    size_t pattern_size;
};

struct urCommandBufferFillCommandsTest
    : uur::command_buffer::urCommandBufferExpTestWithParam<testParametersFill> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::command_buffer::urCommandBufferExpTestWithParam<
                testParametersFill>::SetUp());

        size = std::get<1>(GetParam()).size;
        pattern_size = std::get<1>(GetParam()).pattern_size;
        pattern = std::vector<uint8_t>(pattern_size);
        uur::generateMemFillPattern(pattern);

        // Allocate USM pointers
        ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, size,
                                        &device_ptr));
        ASSERT_NE(device_ptr, nullptr);

        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                         nullptr, &buffer));

        ASSERT_NE(buffer, nullptr);
    }

    void TearDown() override {
        if (device_ptr) {
            EXPECT_SUCCESS(urUSMFree(context, device_ptr));
        }

        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }

        UUR_RETURN_ON_FATAL_FAILURE(
            uur::command_buffer::urCommandBufferExpTestWithParam<
                testParametersFill>::TearDown());
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

    static constexpr unsigned elements = 16;
    static constexpr size_t allocation_size = elements * sizeof(uint32_t);

    std::vector<uint8_t> pattern;
    size_t size;
    size_t pattern_size;

    ur_exp_command_buffer_sync_point_t sync_point;
    void *device_ptr = nullptr;
    ur_mem_handle_t buffer = nullptr;
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

template <typename T>
static std::string
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

UUR_DEVICE_TEST_SUITE_P(urCommandBufferFillCommandsTest,
                        testing::ValuesIn(test_cases),
                        printFillTestString<urCommandBufferFillCommandsTest>);

TEST_P(urCommandBufferFillCommandsTest, Buffer) {
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferFillExp(
        cmd_buf_handle, buffer, pattern.data(), pattern_size, 0, size, 0,
        nullptr, 0, nullptr, &sync_point, nullptr, nullptr));

    std::vector<uint8_t> output(size, 1);
    ASSERT_SUCCESS(urCommandBufferAppendMemBufferReadExp(
        cmd_buf_handle, buffer, 0, size, output.data(), 1, &sync_point, 0,
        nullptr, nullptr, nullptr, nullptr));

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    verifyData(output, size);
}

TEST_P(urCommandBufferFillCommandsTest, USM) {
    ASSERT_SUCCESS(urCommandBufferAppendUSMFillExp(
        cmd_buf_handle, device_ptr, pattern.data(), pattern_size, size, 0,
        nullptr, 0, nullptr, &sync_point, nullptr, nullptr));

    std::vector<uint8_t> output(size, 1);
    ASSERT_SUCCESS(urCommandBufferAppendUSMMemcpyExp(
        cmd_buf_handle, output.data(), device_ptr, size, 1, &sync_point, 0,
        nullptr, nullptr, nullptr, nullptr));

    ASSERT_SUCCESS(urCommandBufferFinalizeExp(cmd_buf_handle));

    ASSERT_SUCCESS(
        urCommandBufferEnqueueExp(cmd_buf_handle, queue, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    verifyData(output, size);
}
