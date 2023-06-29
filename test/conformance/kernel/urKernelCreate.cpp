// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelCreateTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        auto kernel_names =
            uur::KernelsEnvironment::instance->GetEntryPointNames(
                this->program_name);
        kernel_name = kernel_names[0];
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
    std::string invalid_name = "incorrect_kernel_name";
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_NAME,
                     urKernelCreate(program, invalid_name.data(), &kernel));
}
