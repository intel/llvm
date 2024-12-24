// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"
#include <uur/fixtures.h>

struct urUSMSharedAllocTest
    : uur::urQueueTestWithParam<uur::USMDeviceAllocParams> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urQueueTestWithParam<uur::USMDeviceAllocParams>::SetUp());
        ur_device_usm_access_capability_flags_t shared_usm_cross = 0;
        ur_device_usm_access_capability_flags_t shared_usm_single = 0;

        ASSERT_SUCCESS(
            uur::GetDeviceUSMCrossSharedSupport(device, shared_usm_cross));
        ASSERT_SUCCESS(
            uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));

        if (!(shared_usm_cross || shared_usm_single)) {
            GTEST_SKIP() << "Shared USM is not supported by the device.";
        }

        if (usePool) {
            ur_bool_t poolSupport = false;
            ASSERT_SUCCESS(uur::GetDeviceUSMPoolSupport(device, poolSupport));
            if (!poolSupport) {
                GTEST_SKIP() << "USM pools are not supported.";
            }
            ur_usm_pool_desc_t pool_desc = {};
            ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));
        }
    }

    void TearDown() override {
        if (pool) {
            ASSERT_SUCCESS(urUSMPoolRelease(pool));
        }
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urQueueTestWithParam<uur::USMDeviceAllocParams>::TearDown());
    }

    ur_usm_pool_handle_t pool = nullptr;
    bool usePool = std::get<0>(getParam()).value;
    void *ptr = nullptr;
};

// The 0 value parameters are not relevant for urUSMSharedAllocTest tests, they
// are used below in urUSMSharedAllocAlignmentTest for allocation size and
// alignment values
UUR_TEST_SUITE_P(
    urUSMSharedAllocTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(0), testing::Values(0)),
    uur::printUSMAllocTestString<urUSMSharedAllocTest>);

TEST_P(urUSMSharedAllocTest, Success) {
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, pool,
                                    allocation_size, &ptr));
    ASSERT_NE(ptr, nullptr);

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMSharedAllocTest, SuccessWithDescriptors) {
    ur_usm_device_desc_t usm_device_desc{UR_STRUCTURE_TYPE_USM_DEVICE_DESC,
                                         nullptr,
                                         /* device flags */ 0};

    ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC,
                                     &usm_device_desc,
                                     /* host flags */ 0};

    ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                           /* mem advice flags */ UR_USM_ADVICE_FLAG_DEFAULT,
                           /* alignment */ 0};
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, &usm_desc, pool,
                                    allocation_size, &ptr));
    ASSERT_NE(ptr, nullptr);

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMSharedAllocTest, SuccessWithMultipleAdvices) {
    ur_usm_desc_t usm_desc{
        UR_STRUCTURE_TYPE_USM_DESC, nullptr,
        /* mem advice flags */ UR_USM_ADVICE_FLAG_SET_READ_MOSTLY |
            UR_USM_ADVICE_FLAG_BIAS_CACHED,
        /* alignment */ 0};
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, &usm_desc, pool,
                                    allocation_size, &ptr));
    ASSERT_NE(ptr, nullptr);

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMSharedAllocTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urUSMSharedAlloc(nullptr, device, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidNullHandleDevice) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urUSMSharedAlloc(context, nullptr, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidNullPtrMem) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urUSMSharedAlloc(context, device, nullptr, pool, sizeof(int), nullptr));
}

TEST_P(urUSMSharedAllocTest, InvalidUSMSize) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_USM_SIZE,
        urUSMSharedAlloc(context, device, nullptr, pool, -1, &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidValueAlignPowerOfTwo) {
    ur_usm_desc_t desc = {};
    desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
    desc.align = 5;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urUSMSharedAlloc(context, device, &desc, pool, sizeof(int), &ptr));
}

using urUSMSharedAllocAlignmentTest = urUSMSharedAllocTest;

UUR_TEST_SUITE_P(
    urUSMSharedAllocAlignmentTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(4, 8, 16, 32, 64), testing::Values(8, 512, 2048)),
    uur::printUSMAllocTestString<urUSMSharedAllocAlignmentTest>);

TEST_P(urUSMSharedAllocAlignmentTest, SuccessAlignedAllocations) {
    uint32_t alignment = std::get<1>(getParam());
    size_t allocation_size = std::get<2>(getParam());

    ur_usm_device_desc_t usm_device_desc{UR_STRUCTURE_TYPE_USM_DEVICE_DESC,
                                         nullptr,
                                         /* device flags */ 0};

    ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC,
                                     &usm_device_desc,
                                     /* host flags */ 0};

    ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                           /* mem advice flags */ UR_USM_ADVICE_FLAG_DEFAULT,
                           alignment};

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, &usm_desc, pool,
                                    allocation_size, &ptr));
    ASSERT_NE(ptr, nullptr);

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}
