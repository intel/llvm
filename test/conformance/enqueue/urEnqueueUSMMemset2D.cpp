// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include "helpers.h"
#include <uur/fixtures.h>

struct urEnqueueUSMMemset2DTestWithParam
    : uur::urQueueTestWithParam<uur::TestParameters2D> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::SetUp());

        pitch = std::get<1>(GetParam()).pitch;
        width = std::get<1>(GetParam()).width;
        height = std::get<1>(GetParam()).height;
        num_elements = pitch * height;
        host_mem = std::vector<char>(num_elements);

        bool device_usm{false};
        ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT,
                                       sizeof(bool), &device_usm, nullptr));

        if (!device_usm) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, num_elements, 0,
                                        reinterpret_cast<void **>(&ptr)));
    }

    void TearDown() override {
        if (ptr) {
            EXPECT_SUCCESS(urUSMFree(context, ptr));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::TearDown());
    }

    void verifyData() {
        ASSERT_SUCCESS(
            urEnqueueUSMMemcpy2D(queue, true, host_mem.data(), pitch, ptr,
                                 pitch, width, height, 0, nullptr, nullptr));
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                char *host_ptr = host_mem.data();
                size_t index = (pitch * h) + w;
                ASSERT_TRUE((*(host_ptr + index) == memset_value));
            }
        }
    }

    const int memset_value = 12;
    size_t pitch;
    size_t width;
    size_t height;
    size_t num_elements;
    std::vector<char> host_mem;
    int *ptr{nullptr};
};

static std::vector<uur::TestParameters2D> test_cases{
    /* Everything set to 1 */
    {1, 1, 1},
    /* Height == 1 && Pitch > width */
    {1024, 256, 1},
    /* Height == 1 && Pitch == width */
    {1024, 1024, 1},
    /* Height > 1 && Pitch > width */
    {1024, 256, 256},
    /* Height > 1 && Pitch == width + 1 */
    {234, 233, 23},
    /* Height == 1 && Pitch == width + 1 */
    {234, 233, 1}};

UUR_TEST_SUITE_P(urEnqueueUSMMemset2DTestWithParam,
                 testing::ValuesIn(test_cases),
                 uur::print2DTestString<urEnqueueUSMMemset2DTestWithParam>);

TEST_P(urEnqueueUSMMemset2DTestWithParam, Success) {

    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEnqueueUSMMemset2D(queue, ptr, pitch, memset_value, width, height, 0,
                             nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));

    ASSERT_SUCCESS(urEventWait(1, &event));
    ur_event_status_t event_status;
    EXPECT_SUCCESS(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(ur_event_status_t), &event_status,
                                  nullptr));
    ASSERT_EQ(UR_EVENT_STATUS_COMPLETE, event_status);
    EXPECT_SUCCESS(urEventRelease(event));

    ASSERT_NO_FATAL_FAILURE(verifyData());
}

struct urEnqueueUSMMemset2DNegativeTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

        bool device_usm{false};
        ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT,
                                       sizeof(bool), &device_usm, nullptr));

        if (!device_usm) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, num_elements, 0,
                                        reinterpret_cast<void **>(&ptr)));
    }

    void TearDown() override {
        if (ptr) {
            EXPECT_SUCCESS(urUSMFree(context, ptr));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    const int memset_value = 12;
    size_t default_pitch = 16;
    size_t default_width = 16;
    size_t default_height = 16;
    size_t num_elements = default_pitch * default_height;
    int *ptr{nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueUSMMemset2DNegativeTest);

TEST_P(urEnqueueUSMMemset2DNegativeTest, InvalidNullQueueHandle) {
    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(nullptr, ptr, default_pitch, memset_value,
                             default_width, default_height, 0, nullptr,
                             nullptr),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueUSMMemset2DNegativeTest, InvalidNullPtr) {
    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(queue, nullptr, default_pitch, memset_value,
                             default_width, default_height, 0, nullptr,
                             nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueUSMMemset2DNegativeTest, InvalidPitch) {

    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(queue, ptr, 0, memset_value, default_width,
                             default_height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);

    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(queue, ptr, default_width - 1, memset_value,
                             default_width, default_height, 0, nullptr,
                             nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMMemset2DNegativeTest, InvalidWidth) {

    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(queue, ptr, default_pitch, memset_value, 0,
                             default_height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMMemset2DNegativeTest, InvalidHeight) {

    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(queue, ptr, default_pitch, memset_value,
                             default_width, 0, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMMemset2DNegativeTest, OutOfBounds) {

    size_t out_of_bounds = default_pitch * default_height + 1;

    /* Interpret memory as having just one row */
    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(queue, ptr, out_of_bounds, memset_value,
                             default_width, 1, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);

    /* Interpret memory as having just one column */
    ASSERT_EQ_RESULT(
        urEnqueueUSMMemset2D(queue, ptr, 1, memset_value, 1, out_of_bounds,
                             0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMMemset2DNegativeTest, InvalidNullPtrEventWaitList) {

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueUSMMemset2D(queue, ptr, default_pitch,
                                          memset_value, default_width,
                                          default_height, 1, nullptr, nullptr));

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueUSMMemset2D(queue, ptr, default_pitch,
                                          memset_value, default_width,
                                          default_height, 0, &validEvent,
                                          nullptr));
}
