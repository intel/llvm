// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urUSMDeviceAllocTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMDeviceAllocTest);

TEST_P(urUSMDeviceAllocTest, Success) {
    void *ptr = nullptr;
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, sizeof(int), 0, &ptr));
    ASSERT_NE(ptr, nullptr);

    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEnqueueUSMMemset(queue, ptr, 0, sizeof(int), 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMDeviceAllocTest, InvalidNullHandleContext) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE, urUSMDeviceAlloc(nullptr, device, nullptr, nullptr, sizeof(int), 0, &ptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidNullHandleDevice) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_DEVICE, urUSMDeviceAlloc(context, nullptr, nullptr, nullptr, sizeof(int), 0, &ptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidNullPtrResult) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER, urUSMDeviceAlloc(context, device, nullptr, nullptr, sizeof(int), 0, nullptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidUSMSize) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_USM_SIZE,
                     urUSMDeviceAlloc(context, device, nullptr, nullptr, 13, 0, &ptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidValueAlignPowerOfTwo) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urUSMDeviceAlloc(context, device, nullptr, nullptr, sizeof(int), 1, &ptr));
}
