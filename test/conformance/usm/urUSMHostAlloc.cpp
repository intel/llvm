// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"
#include <cstring>
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urUSMHostAllocTest
    : uur::urQueueTestWithParam<uur::USMDeviceAllocParams> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urQueueTestWithParam<uur::USMDeviceAllocParams>::SetUp());
        ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
        ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
        if (!hostUSMSupport) {
            GTEST_SKIP() << "Device USM is not supported.";
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
};

// The 0 value parameters are not relevant for urUSMHostAllocTest tests, they
// are used below in urUSMHostAllocAlignmentTest for allocation size and
// alignment values
UUR_DEVICE_TEST_SUITE_P(
    urUSMHostAllocTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(0), testing::Values(0)),
    uur::printUSMAllocTestString<urUSMHostAllocTest>);

TEST_P(urUSMHostAllocTest, Success) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
    if (!hostUSMSupport) {
        GTEST_SKIP() << "Host USM is not supported.";
    }

    size_t allocation_size = sizeof(int);
    int *ptr = nullptr;
    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, pool, sizeof(int),
                                  reinterpret_cast<void **>(&ptr)));
    ASSERT_NE(ptr, nullptr);

    // Set 0
    ur_event_handle_t event = nullptr;

    uint8_t pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));
    EXPECT_SUCCESS(urEventRelease(event));
    ASSERT_EQ(*ptr, 0);

    // Set 1, in all bytes of int
    pattern = 1;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &event));
    EXPECT_SUCCESS(urEventRelease(event));
    // replicate it on host
    int set_data = 0;
    std::memset(&set_data, 1, allocation_size);
    ASSERT_EQ(*ptr, set_data);

    ASSERT_SUCCESS(urUSMFree(context, ptr));
}

TEST_P(urUSMHostAllocTest, SuccessWithDescriptors) {
    ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC, nullptr,
                                     /* host flags */ 0};

    ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                           /* mem advice flags */ UR_USM_ADVICE_FLAG_DEFAULT,
                           /* alignment */ 0};
    void *ptr = nullptr;
    size_t allocation_size = sizeof(int);
    ASSERT_SUCCESS(
        urUSMHostAlloc(context, &usm_desc, pool, allocation_size, &ptr));
    ASSERT_NE(ptr, nullptr);

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMHostAllocTest, InvalidNullHandleContext) {
    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urUSMHostAlloc(nullptr, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMHostAllocTest, InvalidNullPtrMem) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urUSMHostAlloc(context, nullptr, pool, sizeof(int), nullptr));
}

TEST_P(urUSMHostAllocTest, InvalidUSMSize) {
    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::NativeCPU{});

    void *ptr = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_USM_SIZE,
                     urUSMHostAlloc(context, nullptr, pool, -1, &ptr));
}

TEST_P(urUSMHostAllocTest, InvalidValueAlignPowerOfTwo) {
    void *ptr = nullptr;
    ur_usm_desc_t desc = {};
    desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
    desc.align = 5;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                     urUSMHostAlloc(context, &desc, pool, sizeof(int), &ptr));
}

using urUSMHostAllocAlignmentTest = urUSMHostAllocTest;

UUR_DEVICE_TEST_SUITE_P(
    urUSMHostAllocAlignmentTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(4, 8, 16, 32, 64), testing::Values(8, 512, 2048)),
    uur::printUSMAllocTestString<urUSMHostAllocAlignmentTest>);

TEST_P(urUSMHostAllocAlignmentTest, SuccessAlignedAllocations) {
    uint32_t alignment = std::get<1>(getParam());
    size_t allocation_size = std::get<2>(getParam());

    ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC, nullptr,
                                     /* host flags */ 0};

    ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                           /* mem advice flags */ UR_USM_ADVICE_FLAG_DEFAULT,
                           alignment};

    void *ptr = nullptr;
    ASSERT_SUCCESS(
        urUSMHostAlloc(context, &usm_desc, pool, allocation_size, &ptr));
    ASSERT_NE(ptr, nullptr);

    ur_event_handle_t event = nullptr;
    uint8_t pattern = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                    allocation_size, 0, nullptr, &event));
    ASSERT_SUCCESS(urEventWait(1, &event));

    ASSERT_SUCCESS(urUSMFree(context, ptr));
    EXPECT_SUCCESS(urEventRelease(event));
}
