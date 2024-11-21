// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urContextGetInfoTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextGetInfoTest);

TEST_P(urContextGetInfoTest, SuccessNumDevices) {
    ur_context_info_t info_type = UR_CONTEXT_INFO_NUM_DEVICES;
    size_t size = 0;

    ASSERT_SUCCESS(urContextGetInfo(context, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(uint32_t));

    uint32_t nDevices = 0;
    ASSERT_SUCCESS(
        urContextGetInfo(context, info_type, size, &nDevices, nullptr));

    ASSERT_EQ(nDevices, 1);
}

TEST_P(urContextGetInfoTest, SuccessDevices) {
    ur_context_info_t info_type = UR_CONTEXT_INFO_DEVICES;
    size_t size = 0;

    ASSERT_SUCCESS(urContextGetInfo(context, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);

    ur_device_handle_t queried_device = nullptr;
    ASSERT_SUCCESS(
        urContextGetInfo(context, info_type, size, &queried_device, nullptr));

    size_t devices_count = size / sizeof(ur_device_handle_t);
    ASSERT_EQ(devices_count, 1);
    ASSERT_EQ(queried_device, device);

    for (uint32_t i = 0; i < devices_count; i++) {
        auto &devices = uur::DevicesEnvironment::instance->devices;
        auto queried_device =
            std::find(devices.begin(), devices.end(), devices[i]);
        EXPECT_TRUE(queried_device != devices.end())
            << "device associated with the context is not valid";
    }
}

TEST_P(urContextGetInfoTest, SuccessUSMMemCpy2DSupport) {
    ur_context_info_t info_type = UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT;
    size_t size = 0;

    ASSERT_SUCCESS(urContextGetInfo(context, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(ur_bool_t));
}

TEST_P(urContextGetInfoTest, SuccessUSMFill2DSupport) {
    ur_context_info_t info_type = UR_CONTEXT_INFO_USM_FILL2D_SUPPORT;
    size_t size = 0;

    ASSERT_SUCCESS(urContextGetInfo(context, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(ur_bool_t));
}

TEST_P(urContextGetInfoTest, SuccessReferenceCount) {
    ur_context_info_t info_type = UR_CONTEXT_INFO_REFERENCE_COUNT;
    size_t size = 0;

    ASSERT_SUCCESS(urContextGetInfo(context, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, sizeof(uint32_t));

    uint32_t reference_count = 0;
    ASSERT_SUCCESS(
        urContextGetInfo(context, info_type, size, &reference_count, nullptr));
    ASSERT_GT(reference_count, 0U);
}

TEST_P(urContextGetInfoTest, SuccessAtomicMemoryOrderCapabilities) {
    ur_context_info_t info_type =
        UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES;
    size_t size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urContextGetInfo(context, info_type, 0, nullptr, &size), info_type);
    ASSERT_EQ(size, sizeof(ur_memory_order_capability_flags_t));

    ur_memory_order_capability_flags_t flags = 0;
    ASSERT_SUCCESS(urContextGetInfo(context, info_type, size, &flags, nullptr));

    ASSERT_EQ(flags & UR_MEMORY_ORDER_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urContextGetInfoTest, SuccessAtomicMemoryScopeCapabilities) {
    ur_context_info_t info_type =
        UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES;
    size_t size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urContextGetInfo(context, info_type, 0, nullptr, &size), info_type);
    ASSERT_EQ(size, sizeof(ur_memory_scope_capability_flags_t));

    ur_memory_scope_capability_flags_t flags = 0;
    ASSERT_SUCCESS(urContextGetInfo(context, info_type, size, &flags, nullptr));

    ASSERT_EQ(flags & UR_MEMORY_SCOPE_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urContextGetInfoTest, SuccessAtomicFenceOrderCapabilities) {
    ur_context_info_t info_type =
        UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES;
    size_t size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urContextGetInfo(context, info_type, 0, nullptr, &size), info_type);
    ASSERT_EQ(size, sizeof(ur_memory_order_capability_flags_t));

    ur_memory_order_capability_flags_t flags = 0;
    ASSERT_SUCCESS(urContextGetInfo(context, info_type, size, &flags, nullptr));

    ASSERT_EQ(flags & UR_MEMORY_ORDER_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urContextGetInfoTest, SuccessAtomicFenceScopeCapabilities) {
    ur_context_info_t info_type =
        UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES;
    size_t size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urContextGetInfo(context, info_type, 0, nullptr, &size), info_type);
    ASSERT_EQ(size, sizeof(ur_memory_scope_capability_flags_t));

    ur_memory_scope_capability_flags_t flags = 0;
    ASSERT_SUCCESS(urContextGetInfo(context, info_type, size, &flags, nullptr));

    ASSERT_EQ(flags & UR_MEMORY_SCOPE_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urContextGetInfoTest, InvalidNullHandleContext) {
    uint32_t nDevices = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextGetInfo(nullptr, UR_CONTEXT_INFO_NUM_DEVICES,
                                      sizeof(uint32_t), &nDevices, nullptr));
}

TEST_P(urContextGetInfoTest, InvalidEnumeration) {
    uint32_t nDevices = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urContextGetInfo(context, UR_CONTEXT_INFO_FORCE_UINT32,
                                      sizeof(uint32_t), &nDevices, nullptr));
}

TEST_P(urContextGetInfoTest, InvalidSizePropSize) {
    uint32_t nDevices = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES, 0,
                                      &nDevices, nullptr));
}

TEST_P(urContextGetInfoTest, InvalidSizePropSizeSmall) {
    uint32_t nDevices = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES,
                                      sizeof(nDevices) - 1, &nDevices,
                                      nullptr));
}

TEST_P(urContextGetInfoTest, InvalidNullPointerPropValue) {
    uint32_t nDevices = 0;
    ASSERT_EQ_RESULT(urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES,
                                      sizeof(nDevices), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urContextGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES, 0,
                                      nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
