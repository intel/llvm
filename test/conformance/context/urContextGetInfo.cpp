// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <map>
#include <uur/fixtures.h>

struct urContextGetInfoTestWithInfoParam
    : uur::urContextTestWithParam<ur_context_info_t> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urContextTestWithParam<ur_context_info_t>::SetUp());

        ctx_info_size_map = {
            {UR_CONTEXT_INFO_NUM_DEVICES, sizeof(uint32_t)},
            {UR_CONTEXT_INFO_DEVICES,
             sizeof(ur_device_handle_t) *
                 uur::DevicesEnvironment::instance->devices.size()},
            {UR_CONTEXT_INFO_REFERENCE_COUNT, sizeof(uint32_t)},
            {UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT, sizeof(bool)},
            {UR_CONTEXT_INFO_USM_FILL2D_SUPPORT, sizeof(bool)},
        };
    }

    void TearDown() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urContextTestWithParam<ur_context_info_t>::TearDown());
    }

    std::unordered_map<ur_context_info_t, size_t> ctx_info_size_map;
};

UUR_TEST_SUITE_P(urContextGetInfoTestWithInfoParam,
                 ::testing::Values(

                     UR_CONTEXT_INFO_NUM_DEVICES,          //
                     UR_CONTEXT_INFO_DEVICES,              //
                     UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT, //
                     UR_CONTEXT_INFO_USM_FILL2D_SUPPORT    //

                     ),
                 uur::deviceTestWithParamPrinter<ur_context_info_t>);

TEST_P(urContextGetInfoTestWithInfoParam, Success) {
    ur_context_info_t info = getParam();
    size_t info_size = 0;
    ASSERT_SUCCESS(urContextGetInfo(context, info, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    if (const auto expected_size = ctx_info_size_map.find(info);
        expected_size != ctx_info_size_map.end()) {
        ASSERT_EQ(expected_size->second, info_size);
    }

    std::vector<uint8_t> info_data(info_size);
    ASSERT_SUCCESS(
        urContextGetInfo(context, info, info_size, info_data.data(), nullptr));
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
