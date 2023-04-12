// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urProgramGetInfoTest = uur::urProgramTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramGetInfoTest);

TEST_P(urProgramGetInfoTest, Success) {
    uint32_t ref_count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_REFERENCE_COUNT,
                                    sizeof(ref_count), &ref_count, nullptr));
    ASSERT_NE(0, ref_count);
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
