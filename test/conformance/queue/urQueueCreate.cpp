// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueCreateTest);

TEST_P(urQueueCreateTest, Success) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));
    ASSERT_NE(nullptr, queue);
    ASSERT_SUCCESS(urQueueRelease(queue));
}

TEST_P(urQueueCreateTest, InvalidNullHandleContext) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueCreate(nullptr, device, nullptr, &queue));
}

TEST_P(urQueueCreateTest, InvalidNullHandleDevice) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueCreate(context, nullptr, nullptr, &queue));
}

TEST_P(urQueueCreateTest, InvalidNullPointerQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urQueueCreate(context, device, 0, nullptr));
}

TEST_P(urQueueCreateTest, InvalidValueProperties) {
    ur_queue_handle_t queue = nullptr;
    ur_queue_property_t props[] = {UR_QUEUE_PROPERTIES_FLAGS,
                                   UR_QUEUE_FLAG_FORCE_UINT32, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                     urQueueCreate(context, device, props, &queue));
}

// TODO - test UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES
