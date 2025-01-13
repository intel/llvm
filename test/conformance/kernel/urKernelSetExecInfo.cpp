// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urKernelSetExecInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetExecInfoTest);

TEST_P(urKernelSetExecInfoTest, SuccessIndirectAccess) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

    bool property_value = false;
    ASSERT_SUCCESS(
        urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,
                            sizeof(property_value), nullptr, &property_value));
}

TEST_P(urKernelSetExecInfoTest, InvalidNullHandleKernel) {
    bool property_value = false;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urKernelSetExecInfo(nullptr, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,
                            sizeof(property_value), nullptr, &property_value));
}

TEST_P(urKernelSetExecInfoTest, InvalidEnumeration) {
    bool property_value = false;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_ENUMERATION,
        urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_FORCE_UINT32,
                            sizeof(property_value), nullptr, &property_value));
}

TEST_P(urKernelSetExecInfoTest, InvalidNullPointerPropValue) {
    bool property_value = false;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,
                            sizeof(property_value), nullptr, nullptr));
}

struct urKernelSetExecInfoUSMPointersTest : uur::urKernelTest {
    void SetUp() {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }

    void TearDown() {
        if (allocation) {
            ASSERT_SUCCESS(urUSMFree(context, allocation));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::TearDown());
    }

    size_t allocation_size = 16;
    void *allocation = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetExecInfoUSMPointersTest);

TEST_P(urKernelSetExecInfoUSMPointersTest, SuccessHost) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

    ur_device_usm_access_capability_flags_t host_supported = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_supported));
    if (!host_supported) {
        GTEST_SKIP() << "Host USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                  &allocation));
    ASSERT_NE(allocation, nullptr);
    void *pointers[] = {allocation};

    ASSERT_SUCCESS(urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_USM_PTRS,
                                       sizeof(pointers), nullptr, pointers));
}

TEST_P(urKernelSetExecInfoUSMPointersTest, SuccessDevice) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

    ur_device_usm_access_capability_flags_t device_supported = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_supported));
    if (!device_supported) {
        GTEST_SKIP() << "Device USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &allocation));
    ASSERT_NE(allocation, nullptr);
    void *pointers[] = {allocation};

    ASSERT_SUCCESS(urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_USM_PTRS,
                                       sizeof(pointers), nullptr, pointers));
}

TEST_P(urKernelSetExecInfoUSMPointersTest, SuccessShared) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

    ur_device_usm_access_capability_flags_t shared_supported = 0;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_supported));
    if (!shared_supported) {
        GTEST_SKIP() << "Shared USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &allocation));
    ASSERT_NE(allocation, nullptr);
    void *pointers[] = {allocation};

    ASSERT_SUCCESS(urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_USM_PTRS,
                                       sizeof(pointers), nullptr, pointers));
}

using urKernelSetExecInfoCacheConfigTest =
    uur::urKernelTestWithParam<ur_kernel_cache_config_t>;

UUR_DEVICE_TEST_SUITE_P(
    urKernelSetExecInfoCacheConfigTest,
    ::testing::Values(UR_KERNEL_CACHE_CONFIG_DEFAULT,
                      UR_KERNEL_CACHE_CONFIG_LARGE_SLM,
                      UR_KERNEL_CACHE_CONFIG_LARGE_DATA),
    uur::deviceTestWithParamPrinter<ur_kernel_cache_config_t>);

TEST_P(urKernelSetExecInfoCacheConfigTest, Success) {
    auto property_value = getParam();
    ASSERT_SUCCESS(urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_CACHE_CONFIG,
                                       sizeof(property_value), nullptr,
                                       &property_value));
}
