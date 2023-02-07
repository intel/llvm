// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueCreateTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueCreateTest);

TEST_P(urQueueCreateTest, Success) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
    ASSERT_NE(nullptr, queue);
    ASSERT_SUCCESS(urQueueRelease(queue));
}

TEST_P(urQueueCreateTest, InvalidNullHandleContext) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueCreate(nullptr, device, 0, &queue));
}

TEST_P(urQueueCreateTest, InvalidNullHandleDevice) {
    ur_queue_handle_t queue = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueCreate(context, nullptr, 0, &queue));
}

TEST_P(urQueueCreateTest, InvalidEnumerationProps) {
    ur_queue_handle_t queue = nullptr;
    ur_queue_property_t props[] = {UR_QUEUE_PROPERTIES_FLAGS,
                                   UR_QUEUE_FLAG_FORCE_UINT32, 0};
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urQueueCreate(context, device, props, &queue));
}

TEST_P(urQueueCreateTest, InvalidNullPointerQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urQueueCreate(context, device, 0, nullptr));
}
