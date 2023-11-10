// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urProgramGetBuildInfoTest
    : uur::urProgramTestWithParam<ur_program_build_info_t> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urProgramTestWithParam<ur_program_build_info_t>::SetUp());
        ASSERT_SUCCESS(urProgramBuild(this->context, program, nullptr));
    }
};

UUR_TEST_SUITE_P(urProgramGetBuildInfoTest,
                 ::testing::Values(UR_PROGRAM_BUILD_INFO_STATUS,
                                   UR_PROGRAM_BUILD_INFO_OPTIONS,
                                   UR_PROGRAM_BUILD_INFO_LOG,
                                   UR_PROGRAM_BUILD_INFO_BINARY_TYPE),
                 uur::deviceTestWithParamPrinter<ur_program_build_info_t>);

TEST_P(urProgramGetBuildInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS(urProgramGetBuildInfo(program, device, property_name, 0,
                                         nullptr, &property_size));
    property_value.resize(property_size);
    ASSERT_SUCCESS(urProgramGetBuildInfo(program, device, property_name,
                                         property_size, property_value.data(),
                                         nullptr));
}

TEST_P(urProgramGetBuildInfoTest, InvalidNullHandleProgram) {
    ur_program_build_status_t programBuildStatus =
        UR_PROGRAM_BUILD_STATUS_ERROR;
    ASSERT_EQ_RESULT(urProgramGetBuildInfo(nullptr, device,
                                           UR_PROGRAM_BUILD_INFO_STATUS,
                                           sizeof(programBuildStatus),
                                           &programBuildStatus, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urProgramGetBuildInfoTest, InvalidNullHandleDevice) {
    ur_program_build_status_t programBuildStatus =
        UR_PROGRAM_BUILD_STATUS_ERROR;
    ASSERT_EQ_RESULT(urProgramGetBuildInfo(program, nullptr,
                                           UR_PROGRAM_BUILD_INFO_STATUS,
                                           sizeof(programBuildStatus),
                                           &programBuildStatus, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urProgramGetBuildInfoTest, InvalidEnumeration) {
    size_t propSizeOut = UR_PROGRAM_BUILD_STATUS_ERROR;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urProgramGetBuildInfo(program, device,
                                           UR_PROGRAM_BUILD_INFO_FORCE_UINT32,
                                           0, nullptr, &propSizeOut));
}
