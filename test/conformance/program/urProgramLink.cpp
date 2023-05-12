// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urProgramLinkTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramCompile(context, program, nullptr));
        programs.push_back(program);

        uur::KernelsEnvironment::instance->LoadSource("bar", 0, bar_binary);
        ASSERT_SUCCESS(urProgramCreateWithIL(context, bar_binary->data(),
                                             bar_binary->size(), nullptr,
                                             &program));
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
    size_t original_kernel_count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_NUM_KERNELS,
                                    sizeof(original_kernel_count),
                                    &original_kernel_count, nullptr));

    size_t linked_kernel_count = 0;
    ASSERT_SUCCESS(urProgramGetInfo(linked_program, UR_PROGRAM_INFO_NUM_KERNELS,
                                    sizeof(linked_kernel_count),
                                    &linked_kernel_count, nullptr));

    ASSERT_GT(linked_kernel_count, original_kernel_count);
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
