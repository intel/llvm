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

struct urProgramGetBuildInfoSingleTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(this->context, program, nullptr));
    }
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramGetBuildInfoSingleTest);

TEST_P(urProgramGetBuildInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urProgramGetBuildInfo(program, device, property_name, 0, nullptr,
                              &property_size),
        property_name);
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

TEST_P(urProgramGetBuildInfoSingleTest, LogIsNullTerminated) {
    size_t logSize;
    std::vector<char> log;

    ASSERT_SUCCESS(urProgramGetBuildInfo(
        program, device, UR_PROGRAM_BUILD_INFO_LOG, 0, nullptr, &logSize));
    // The size should always include the null terminator.
    ASSERT_GT(logSize, 0);
    log.resize(logSize);
    log[logSize - 1] = 'x';
    ASSERT_SUCCESS(urProgramGetBuildInfo(program, device,
                                         UR_PROGRAM_BUILD_INFO_LOG, logSize,
                                         log.data(), nullptr));
    ASSERT_EQ(log[logSize - 1], '\0');
}
