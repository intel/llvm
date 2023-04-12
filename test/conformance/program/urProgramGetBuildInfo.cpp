// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urProgramGetBuildInfoTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
    }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramGetBuildInfoTest);

TEST_P(urProgramGetBuildInfoTest, Success) {
    ur_program_build_status_t programBuildStatus =
        UR_PROGRAM_BUILD_STATUS_ERROR;
    ASSERT_SUCCESS(urProgramGetBuildInfo(
        program, device, UR_PROGRAM_BUILD_INFO_STATUS,
        sizeof(programBuildStatus), &programBuildStatus, nullptr));
    ASSERT_NE(UR_PROGRAM_BUILD_STATUS_ERROR, programBuildStatus);
}

TEST_P(urProgramGetBuildInfoTest, InvalidNullHandleProgram) {
    ur_program_build_status_t programBuildStatus =
        UR_PROGRAM_BUILD_STATUS_ERROR;
    ASSERT_SUCCESS(urProgramGetBuildInfo(
        nullptr, device, UR_PROGRAM_BUILD_INFO_STATUS,
        sizeof(programBuildStatus), &programBuildStatus, nullptr));
}

TEST_P(urProgramGetBuildInfoTest, InvalidNullHandleDevice) {
    ur_program_build_status_t programBuildStatus =
        UR_PROGRAM_BUILD_STATUS_ERROR;
    ASSERT_SUCCESS(urProgramGetBuildInfo(
        program, nullptr, UR_PROGRAM_BUILD_INFO_STATUS,
        sizeof(programBuildStatus), &programBuildStatus, nullptr));
}

TEST_P(urProgramGetBuildInfoTest, InvalidEnumeration) {
    size_t propSizeOut = UR_PROGRAM_BUILD_STATUS_ERROR;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urProgramGetBuildInfo(program, device,
                                           UR_PROGRAM_BUILD_INFO_FORCE_UINT32,
                                           0, nullptr, &propSizeOut));
}
