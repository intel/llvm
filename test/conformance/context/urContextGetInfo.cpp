// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

struct urContextGetInfoTestWithInfoParam
    : uur::urContextTestWithParam<ur_context_info_t> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urContextTestWithParam<ur_context_info_t>::SetUp());

        ctx_info_size_map = {
            {UR_CONTEXT_INFO_NUM_DEVICES, sizeof(uint32_t)},
            {UR_CONTEXT_INFO_DEVICES, sizeof(ur_device_handle_t)},
            {UR_CONTEXT_INFO_REFERENCE_COUNT, sizeof(uint32_t)},
            {UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT, sizeof(bool)},
            {UR_CONTEXT_INFO_USM_FILL2D_SUPPORT, sizeof(bool)},
            {UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
             sizeof(ur_memory_order_capability_flags_t)},
            {UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
             sizeof(ur_memory_order_capability_flags_t)},
            {UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES,
             sizeof(ur_memory_order_capability_flags_t)},
            {UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES,
             sizeof(ur_memory_order_capability_flags_t)}};

        ctx_info_mem_flags_map = {
            {UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES,
             UR_MEMORY_ORDER_CAPABILITY_FLAGS_MASK},
            {UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES,
             UR_MEMORY_SCOPE_CAPABILITY_FLAGS_MASK},
            {UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES,
             UR_MEMORY_ORDER_CAPABILITY_FLAGS_MASK},
            {UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES,
             UR_MEMORY_SCOPE_CAPABILITY_FLAGS_MASK},
        };
    }

    void TearDown() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urContextTestWithParam<ur_context_info_t>::TearDown());
    }

    std::unordered_map<ur_context_info_t, size_t> ctx_info_size_map;
    std::unordered_map<ur_context_info_t, ur_memory_order_capability_flags_t>
        ctx_info_mem_flags_map;
};

UUR_TEST_SUITE_P(urContextGetInfoTestWithInfoParam,
                 ::testing::Values(

                     UR_CONTEXT_INFO_NUM_DEVICES,                      //
                     UR_CONTEXT_INFO_DEVICES,                          //
                     UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT,             //
                     UR_CONTEXT_INFO_USM_FILL2D_SUPPORT,               //
                     UR_CONTEXT_INFO_REFERENCE_COUNT,                  //
                     UR_CONTEXT_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES, //
                     UR_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES, //
                     UR_CONTEXT_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES,  //
                     UR_CONTEXT_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES   //
                     ),
                 uur::deviceTestWithParamPrinter<ur_context_info_t>);

TEST_P(urContextGetInfoTestWithInfoParam, Success) {
    ur_context_info_t info = getParam();
    size_t info_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urContextGetInfo(context, info, 0, nullptr, &info_size), info);
    ASSERT_NE(info_size, 0);

    if (const auto expected_size = ctx_info_size_map.find(info);
        expected_size != ctx_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, info_size);
    }

    std::vector<uint8_t> info_data(info_size);
    ASSERT_SUCCESS(
        urContextGetInfo(context, info, info_size, info_data.data(), nullptr));

    switch (info) {
    case UR_CONTEXT_INFO_NUM_DEVICES: {
        auto returned_num_of_devices =
            reinterpret_cast<uint32_t *>(info_data.data());
        ASSERT_GE(uur::DevicesEnvironment::instance->devices.size(),
                  *returned_num_of_devices);
        break;
    }
    case UR_CONTEXT_INFO_DEVICES: {
        auto returned_devices =
            reinterpret_cast<ur_device_handle_t *>(info_data.data());
        size_t devices_count = info_size / sizeof(ur_device_handle_t);
        ASSERT_GT(devices_count, 0);
        for (uint32_t i = 0; i < devices_count; i++) {
            auto &devices = uur::DevicesEnvironment::instance->devices;
            auto queried_device =
                std::find(devices.begin(), devices.end(), returned_devices[i]);
            EXPECT_TRUE(queried_device != devices.end())
                << "device associated with the context is not valid";
        }
        break;
    }
    case UR_CONTEXT_INFO_REFERENCE_COUNT: {
        auto returned_reference_count =
            reinterpret_cast<uint32_t *>(info_data.data());
        ASSERT_GT(*returned_reference_count, 0U);
        break;
    }
    default:
        break;
    }
}

using urContextGetInfoTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextGetInfoTest);
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
