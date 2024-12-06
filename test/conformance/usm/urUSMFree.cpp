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
        urUSMHostAlloc(context, nullptr, nullptr, allocation_size, &ptr));

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

// This goal of this test is to ensure urUSMFree blocks and waits for operations
// accessing the given allocation to finish before actually freeing the memory.
struct urUSMFreeDuringExecutionTest : uur::urKernelExecutionTest {
    void SetUp() {
        program_name = "fill_usm";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    void *allocation = nullptr;
    size_t array_size = 256;
    size_t allocation_size = array_size * sizeof(uint32_t);
    uint32_t data = 42;
    size_t wg_offset = 0;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urUSMFreeDuringExecutionTest);

TEST_P(urUSMFreeDuringExecutionTest, SuccessHost) {
    ur_device_usm_access_capability_flags_t host_usm_flags = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_usm_flags));
    if (!(host_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
        GTEST_SKIP() << "Host USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                  &allocation));
    ASSERT_NE(allocation, nullptr);

    EXPECT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, allocation));
    EXPECT_SUCCESS(
        urKernelSetArgValue(kernel, 1, sizeof(data), nullptr, &data));
    EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, 1, &wg_offset,
                                         &array_size, nullptr, 0, nullptr,
                                         nullptr));
    ASSERT_SUCCESS(urUSMFree(context, allocation));
    ASSERT_SUCCESS(urQueueFinish(queue));
}

TEST_P(urUSMFreeDuringExecutionTest, SuccessDevice) {
    ur_device_usm_access_capability_flags_t device_usm_flags = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm_flags));
    if (!(device_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
        GTEST_SKIP() << "Device USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &allocation));
    ASSERT_NE(allocation, nullptr);

    EXPECT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, allocation));
    EXPECT_SUCCESS(
        urKernelSetArgValue(kernel, 1, sizeof(data), nullptr, &data));

    EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, 1, &wg_offset,
                                         &array_size, nullptr, 0, nullptr,
                                         nullptr));
    ASSERT_SUCCESS(urUSMFree(context, allocation));
    ASSERT_SUCCESS(urQueueFinish(queue));
}

TEST_P(urUSMFreeDuringExecutionTest, SuccessShared) {
    ur_device_usm_access_capability_flags_t shared_usm_flags = 0;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
        GTEST_SKIP() << "Shared USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &allocation));
    ASSERT_NE(allocation, nullptr);

    EXPECT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, allocation));
    EXPECT_SUCCESS(
        urKernelSetArgValue(kernel, 1, sizeof(data), nullptr, &data));
    EXPECT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, 1, &wg_offset,
                                         &array_size, nullptr, 0, nullptr,
                                         nullptr));
    ASSERT_SUCCESS(urUSMFree(context, allocation));
    ASSERT_SUCCESS(urQueueFinish(queue));
}
