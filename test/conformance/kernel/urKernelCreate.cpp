// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urKernelCreateTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        size_t kernel_string_size = 0;
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_KERNEL_NAMES,
                                        0, nullptr, &kernel_string_size));
        std::string kernel_string;
        kernel_string.resize(kernel_string_size);
        ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_KERNEL_NAMES,
                                        kernel_string.size(),
                                        kernel_string.data(), nullptr));
        kernel_name = kernel_string.substr(0, kernel_string.find(";"));
    }

    void TearDown() override {
        if (kernel) {
            ASSERT_SUCCESS(urKernelRelease(kernel));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::TearDown());
    }

    std::string kernel_name;
    ur_kernel_handle_t kernel = nullptr;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelCreateTest);

TEST_P(urKernelCreateTest, Success) {
    ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));
}

TEST_P(urKernelCreateTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelCreate(nullptr, kernel_name.data(), &kernel));
}

TEST_P(urKernelCreateTest, InvalidNullPointerName) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urKernelCreate(program, nullptr, &kernel));
}

TEST_P(urKernelCreateTest, InvalidNullPointerKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urKernelCreate(program, kernel_name.data(), nullptr));
}

TEST_P(urKernelCreateTest, InvalidKernelName) {
    std::string invalid_name = "";
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_NAME,
                     urKernelCreate(program, invalid_name.data(), &kernel));
}
