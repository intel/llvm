// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urProgramGetInfoTest = uur::urProgramTestWithParam<ur_program_info_t>;

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
    ASSERT_SUCCESS(urProgramGetInfo(nullptr, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                    sizeof(ref_count), &ref_count, nullptr));
}

TEST_P(urProgramGetInfoTest, InvalidEnumeration) {
    size_t prop_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urProgramGetInfo(program, UR_PROGRAM_INFO_FORCE_UINT32, 0,
                                      nullptr, &prop_size));
}
