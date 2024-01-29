// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urUSMFreeTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMFreeTest);

TEST_P(urUSMFreeTest, SuccessDeviceAlloc) {
    ur_device_usm_access_capability_flags_t deviceUSMSupport = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSMSupport));
    if (!deviceUSMSupport) {
        GTEST_SKIP() << "Device USM not supported.";
    }

    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &ptr));

    ur_event_handle_t event = nullptr;

    uint8_t pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(urUSMFree(context, ptr));
    ASSERT_SUCCESS(urEventRelease(event));
}
TEST_P(urUSMFreeTest, SuccessHostAlloc) {
    ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, hostUSMSupport));
    if (!hostUSMSupport) {
        GTEST_SKIP() << "Host USM not supported.";
    }

    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(
        urUSMHostAlloc(context, nullptr, nullptr, sizeof(int), &ptr));

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(urUSMFree(context, ptr));
    ASSERT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMFreeTest, SuccessSharedAlloc) {
    ur_device_usm_access_capability_flags_t shared_usm_cross = 0;
    ur_device_usm_access_capability_flags_t shared_usm_single = 0;

    ASSERT_SUCCESS(
        uur::GetDeviceUSMCrossSharedSupport(device, shared_usm_cross));
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));

    if (!(shared_usm_cross || shared_usm_single)) {
        GTEST_SKIP() << "Shared USM is not supported by the device.";
    }

    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &ptr));

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(urUSMFree(context, ptr));
    ASSERT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMFreeTest, InvalidNullContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMFree(nullptr, nullptr));
}

TEST_P(urUSMFreeTest, InvalidNullPtrMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urUSMFree(context, nullptr));
}
