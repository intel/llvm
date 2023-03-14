// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urUSMFreeTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMFreeTest);

TEST_P(urUSMFreeTest, SuccessDeviceAlloc) {
    bool deviceUSMSupport = false;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSMSupport));
    if (!deviceUSMSupport) {
        GTEST_SKIP() << "Device USM not supported.";
    }

    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, allocation_size, 0,
                         &ptr));

    ur_event_handle_t event = nullptr;

    uint8_t pattern = 0;
    ASSERT_SUCCESS(
        urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern, allocation_size,
                         0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(urUSMFree(context, ptr));
}
TEST_P(urUSMFreeTest, SuccessHostAlloc) {
    bool hostUSMSupport = false;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, hostUSMSupport));
    if (!hostUSMSupport) {
        GTEST_SKIP() << "Host USM not supported.";
    }

    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(
        urUSMHostAlloc(context, nullptr, nullptr, allocation_size, 0, &ptr));

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    ASSERT_SUCCESS(
        urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern, allocation_size,
                         0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(urUSMFree(context, ptr));
}

TEST_P(urUSMFreeTest, SuccessSharedAlloc) {
    bool shared_usm_cross = false;
    bool shared_usm_single = false;

    ASSERT_SUCCESS(uur::GetDeviceUSMCrossSharedSupport(device, shared_usm_cross));
    ASSERT_SUCCESS(uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));

    if (!(shared_usm_cross || shared_usm_single)) {
        GTEST_SKIP() << "Shared USM is not supported by the device.";
    }

    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(
        urUSMSharedAlloc(context, device, nullptr, nullptr, allocation_size, 0,
                         &ptr));

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    ASSERT_SUCCESS(
        urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern, allocation_size,
                         0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(urUSMFree(context, ptr));
}

TEST_P(urUSMFreeTest, InvalidNullContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMFree(nullptr, nullptr));
}

TEST_P(urUSMFreeTest, InvalidNullPtrMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urUSMFree(context, nullptr));
}
