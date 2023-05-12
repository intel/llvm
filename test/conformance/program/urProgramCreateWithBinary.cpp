// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urProgramCreateWithBinaryTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        size_t binary_size = 0;
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES, 0,
                                        nullptr, &binary_size));
        binary.resize(binary_size);
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                        binary_size, binary.data(), nullptr));
    }

    void TearDown() override {
        if (binary_program) {
            EXPECT_SUCCESS(urProgramRelease(binary_program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::TearDown());
    }

    std::vector<uint8_t> binary;
    ur_program_handle_t binary_program = nullptr;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramCreateWithBinaryTest);

TEST_P(urProgramCreateWithBinaryTest, Success) {
    ASSERT_SUCCESS(urProgramCreateWithBinary(context, device, binary.size(),
                                             binary.data(), nullptr,
                                             &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCreateWithBinary(nullptr, device, binary.size(),
                                               binary.data(), nullptr,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullHandleDevice) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCreateWithBinary(context, nullptr, binary.size(),
                                               binary.data(), nullptr,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullPointerBinary) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithBinary(context, device, binary.size(),
                                               nullptr, nullptr,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullPointerProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithBinary(context, device, binary.size(),
                                               binary.data(), nullptr,
                                               nullptr));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullPointerMetadata) {
    ur_program_properties_t properties = {};
    properties.count = 1;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithBinary(context, device, binary.size(),
                                               binary.data(), &properties,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidSizePropertyCount) {
    std::string md_string = "test metadata";
    ur_program_metadata_value_t md_value = {};
    md_value.pString = md_string.data();
    ur_program_metadata_t md = {md_string.data(),
                                UR_PROGRAM_METADATA_TYPE_STRING,
                                md_string.size(), md_value};
    ur_program_properties_t properties = {};
    properties.pMetadatas = &md;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urProgramCreateWithBinary(context, device, binary.size(),
                                               binary.data(), &properties,
                                               &binary_program));
}
