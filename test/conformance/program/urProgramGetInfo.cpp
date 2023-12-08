// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urProgramGetInfoTest : uur::urProgramTestWithParam<ur_program_info_t> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urProgramTestWithParam<ur_program_info_t>::SetUp());
        // Some queries need the program to be built.
        ASSERT_SUCCESS(urProgramBuild(this->context, program, nullptr));
    }
};

UUR_TEST_SUITE_P(
    urProgramGetInfoTest,
    ::testing::Values(UR_PROGRAM_INFO_REFERENCE_COUNT, UR_PROGRAM_INFO_CONTEXT,
                      UR_PROGRAM_INFO_NUM_DEVICES, UR_PROGRAM_INFO_DEVICES,
                      UR_PROGRAM_INFO_SOURCE, UR_PROGRAM_INFO_BINARY_SIZES,
                      UR_PROGRAM_INFO_BINARIES, UR_PROGRAM_INFO_NUM_KERNELS,
                      UR_PROGRAM_INFO_KERNEL_NAMES),
    uur::deviceTestWithParamPrinter<ur_program_info_t>);

TEST_P(urProgramGetInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS(
        urProgramGetInfo(program, property_name, 0, nullptr, &property_size));
    property_value.resize(property_size);
    ASSERT_SUCCESS(urProgramGetInfo(program, property_name, property_size,
                                    property_value.data(), nullptr));
}

TEST_P(urProgramGetInfoTest, InvalidNullHandleProgram) {
    uint32_t ref_count = 0;
    ASSERT_EQ_RESULT(urProgramGetInfo(nullptr, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      sizeof(ref_count), &ref_count, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urProgramGetInfoTest, InvalidEnumeration) {
    size_t prop_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urProgramGetInfo(program, UR_PROGRAM_INFO_FORCE_UINT32, 0,
                                      nullptr, &prop_size));
}

TEST_P(urProgramGetInfoTest, InvalidSizeZero) {
    uint32_t ref_count = 0;
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      0, &ref_count, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urProgramGetInfoTest, InvalidSizeSmall) {
    uint32_t ref_count = 0;
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      sizeof(ref_count) - 1, &ref_count,
                                      nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urProgramGetInfoTest, InvalidNullPointerPropValue) {
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      sizeof(uint32_t), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urProgramGetInfoTest, InvalidNullPointerPropValueRet) {
    ASSERT_EQ_RESULT(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                      0, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
