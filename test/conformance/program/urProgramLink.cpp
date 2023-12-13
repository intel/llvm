// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urProgramLinkTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        // TODO: This should use a query for urProgramCreateWithIL support or
        // rely on UR_RESULT_ERROR_UNSUPPORTED_FEATURE being returned.
        ur_platform_backend_t backend;
        ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(ur_platform_backend_t),
                                         &backend, nullptr));
        if (backend == UR_PLATFORM_BACKEND_HIP) {
            GTEST_SKIP();
        }
        ASSERT_SUCCESS(urProgramCompile(context, program, nullptr));
        programs.push_back(program);

        uur::KernelsEnvironment::instance->LoadSource("bar", 0, bar_binary);
        ASSERT_SUCCESS(urProgramCreateWithIL(context, bar_binary->data(),
                                             bar_binary->size(), nullptr,
                                             &bar_program));
        ASSERT_SUCCESS(urProgramCompile(context, bar_program, nullptr));
        programs.push_back(bar_program);
    }

    void TearDown() override {
        if (bar_program) {
            EXPECT_SUCCESS(urProgramRelease(bar_program));
        }
        if (linked_program) {
            EXPECT_SUCCESS(urProgramRelease(linked_program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::TearDown());
    }

    ur_program_handle_t bar_program = nullptr;
    ur_program_handle_t linked_program = nullptr;
    std::shared_ptr<std::vector<char>> bar_binary;
    std::vector<ur_program_handle_t> programs;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramLinkTest);

TEST_P(urProgramLinkTest, Success) {
    ASSERT_SUCCESS(urProgramLink(context, programs.size(), programs.data(),
                                 nullptr, &linked_program));
    ur_program_binary_type_t binary_type = UR_PROGRAM_BINARY_TYPE_NONE;
    ASSERT_SUCCESS(urProgramGetBuildInfo(
        program, device, UR_PROGRAM_BUILD_INFO_BINARY_TYPE, sizeof(binary_type),
        &binary_type, nullptr));
}

TEST_P(urProgramLinkTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramLink(nullptr, programs.size(), programs.data(),
                                   nullptr, &linked_program));
}

TEST_P(urProgramLinkTest, InvalidNullPointerProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramLink(context, programs.size(), programs.data(),
                                   nullptr, nullptr));
}

TEST_P(urProgramLinkTest, InvalidNullPointerInputPrograms) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramLink(context, programs.size(), nullptr, nullptr,
                                   &linked_program));
}

TEST_P(urProgramLinkTest, InvalidSizeCount) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_SIZE,
        urProgramLink(context, 0, programs.data(), nullptr, &linked_program));
}
