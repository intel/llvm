// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urUSMFreeTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMFreeTest);

TEST_P(urUSMFreeTest, SuccessDeviceAlloc) {
    const auto deviceUSMSupport =
        uur::GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT);
    ASSERT_TRUE(deviceUSMSupport.has_value());
    if (!deviceUSMSupport.value()) {
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
    const auto hostUSMSupport =
        uur::GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_HOST_SUPPORT);
    ASSERT_TRUE(hostUSMSupport.has_value());
    if (!hostUSMSupport.value()) {
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
    auto sharedUSMCross = uur::GetDeviceInfo<bool>(
        device, UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT);
    auto sharedUSMSingle = uur::GetDeviceInfo<bool>(
        device, UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT);

    ASSERT_TRUE(sharedUSMCross.has_value() && sharedUSMSingle.has_value());

    if (!(sharedUSMCross.value() || !sharedUSMCross.value())) {
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
