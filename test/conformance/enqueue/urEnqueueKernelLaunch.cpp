// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urEnqueueKernelLaunchTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size = 32;
    size_t global_offset = 0;
    size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunchTest);

TEST_P(urEnqueueKernelLaunchTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(sizeof(val) * global_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size, nullptr,
                                         0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(nullptr, kernel, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, nullptr, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    // does this make sense??
    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                           &global_offset, &global_size,
                                           nullptr, 0, &validEvent, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}

TEST_P(urEnqueueKernelLaunchTest, InvalidWorkDimension) {
    uint32_t max_work_item_dimensions = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
        sizeof(max_work_item_dimensions), &max_work_item_dimensions, nullptr));
    ASSERT_EQ_RESULT(urEnqueueKernelLaunch(queue, kernel,
                                           max_work_item_dimensions + 1,
                                           &global_offset, &global_size,
                                           nullptr, 0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
}

struct urEnqueueKernelLaunch2DTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill_2d";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size[2] = {8, 8};
    size_t global_offset[2] = {0, 0};
    size_t buffer_size = sizeof(val) * global_size[0] * global_size[1];
    size_t n_dimensions = 2;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunch2DTest);

TEST_P(urEnqueueKernelLaunch2DTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(buffer_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         global_offset, global_size, nullptr,
                                         0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, buffer_size, val);
}

struct urEnqueueKernelLaunch3DTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill_3d";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size[3] = {4, 4, 4};
    size_t global_offset[3] = {0, 0, 0};
    size_t buffer_size = sizeof(val) * global_size[0] * global_size[1] * global_size[2];
    size_t n_dimensions = 3;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunch3DTest);

TEST_P(urEnqueueKernelLaunch3DTest, Success) {
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(buffer_size, &buffer);
    AddPodArg(val);
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         global_offset, global_size, nullptr,
                                         0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, buffer_size, val);
}
