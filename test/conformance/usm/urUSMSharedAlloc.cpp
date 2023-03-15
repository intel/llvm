// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urUSMSharedAllocTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());
        bool shared_usm_cross = false;
        bool shared_usm_single = false;

        ASSERT_SUCCESS(
            uur::GetDeviceUSMCrossSharedSupport(device, shared_usm_cross));
        ASSERT_SUCCESS(
            uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));

        if (!(shared_usm_cross || shared_usm_single)) {
            GTEST_SKIP() << "Shared USM is not supported by the device.";
        }
    }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMSharedAllocTest);

TEST_P(urUSMSharedAllocTest, Success) {
    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, 0, &ptr));

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMSharedAllocTest, InvalidNullHandleContext) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMSharedAlloc(nullptr, device, nullptr, nullptr,
                                      sizeof(int), 0, &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidNullHandleDevice) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMSharedAlloc(context, nullptr, nullptr, nullptr,
                                      sizeof(int), 0, &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidNullPtrMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      sizeof(int), 0, nullptr));
}

TEST_P(urUSMSharedAllocTest, InvalidUSMSize) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_USM_SIZE,
        urUSMSharedAlloc(context, device, nullptr, nullptr, 13, 0, &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidValueAlignPowerOfTwo) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                     urUSMSharedAlloc(context, device, nullptr, nullptr,
                                      sizeof(int), 1, &ptr));
}
