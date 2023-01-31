// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urContextGetInfoTest = uur::urContextTestWithParam<ur_context_info_t>;

UUR_TEST_SUITE_P(
    urContextGetInfoTest,
    ::testing::Values(

        UR_CONTEXT_INFO_NUM_DEVICES,          //
        UR_CONTEXT_INFO_DEVICES,              //
        UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT, //
        UR_CONTEXT_INFO_USM_FILL2D_SUPPORT,   //
        UR_CONTEXT_INFO_USM_MEMSET2D_SUPPORT  //

        ),
    [](const ::testing::TestParamInfo<urContextGetInfoTest::ParamType> &info) {
        ur_device_handle_t device = std::get<0>(info.param);
        ur_context_info_t context_info = std::get<1>(info.param);

        std::stringstream ss;
        ss << context_info;
        return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
    });

TEST_P(urContextGetInfoTest, Success) {
    ur_context_info_t info = getParam();
    size_t info_size = 0;
    ASSERT_SUCCESS(urContextGetInfo(context, info, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);
    std::vector<uint8_t> info_data(info_size);
    ASSERT_SUCCESS(
        urContextGetInfo(context, info, info_size, info_data.data(), nullptr));
}
