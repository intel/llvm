// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urProgramCreateWithBinaryTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        size_t binary_sizes_len = 0;
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                        0, nullptr, &binary_sizes_len));
        // We're expecting one binary
        ASSERT_EQ(binary_sizes_len / sizeof(size_t), 1);
        size_t binary_size = 0;
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                        sizeof(binary_size), &binary_size,
                                        nullptr));
        binary.resize(binary_size);
        uint8_t *binary_ptr = binary.data();
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                        sizeof(binary_ptr), &binary_ptr,
                                        nullptr));
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
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramCreateWithBinaryTest);

TEST_P(urProgramCreateWithBinaryTest, Success) {
    auto size = binary.size();
    const uint8_t *data = binary.data();
    ASSERT_SUCCESS(urProgramCreateWithBinary(context, 1, &device, &size, &data,
                                             nullptr, &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullHandleContext) {
    auto size = binary.size();
    const uint8_t *data = binary.data();
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramCreateWithBinary(nullptr, 1, &device, &size,
                                               &data, nullptr,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullHandleDevice) {
    auto size = binary.size();
    const uint8_t *data = binary.data();
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithBinary(context, 0, nullptr, &size,
                                               &data, nullptr,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullPointerBinary) {
    auto size = binary.size();
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithBinary(context, 1, &device, &size,
                                               nullptr, nullptr,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullPointerProgram) {
    auto size = binary.size();
    const uint8_t *data = binary.data();
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithBinary(context, 1, &device, &size,
                                               &data, nullptr, nullptr));
}

TEST_P(urProgramCreateWithBinaryTest, InvalidNullPointerMetadata) {
    ur_program_properties_t properties = {};
    properties.count = 1;
    auto size = binary.size();
    const uint8_t *data = binary.data();
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramCreateWithBinary(context, 1, &device, &size,
                                               &data, &properties,
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
    auto size = binary.size();
    const uint8_t *data = binary.data();
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urProgramCreateWithBinary(context, 1, &device, &size,
                                               &data, &properties,
                                               &binary_program));
}

TEST_P(urProgramCreateWithBinaryTest, BuildInvalidProgramBinary) {
    ur_program_handle_t program = nullptr;
    uint8_t binary[] = {0, 1, 2, 3, 4};
    const uint8_t *data = binary;
    size_t size = 5;
    auto result = urProgramCreateWithBinary(context, 1, &device, &size, &data,
                                            nullptr, &program);
    // The driver is not required to reject the binary
    ASSERT_TRUE(result == UR_RESULT_ERROR_INVALID_BINARY ||
                result == UR_RESULT_SUCCESS);
}
