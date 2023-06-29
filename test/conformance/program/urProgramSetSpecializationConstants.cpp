// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urProgramSetSpecializationConstantsTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "spec_constant";
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
    }

    uint32_t spec_value = 42;
    ur_specialization_constant_info_t info = {0, sizeof(spec_value),
                                              &spec_value};
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urProgramSetSpecializationConstantsTest);

TEST_P(urProgramSetSpecializationConstantsTest, Success) {
    ASSERT_SUCCESS(urProgramSetSpecializationConstants(program, 1, &info));
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
    auto entry_points =
        uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
    kernel_name = entry_points[0];
    ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));

    ur_mem_handle_t buffer;
    AddBuffer1DArg(sizeof(spec_value), &buffer);
    Launch1DRange(1);
    ValidateBuffer<uint32_t>(buffer, sizeof(spec_value), spec_value);
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urProgramSetSpecializationConstants(nullptr, 1, &info));
}

TEST_P(urProgramSetSpecializationConstantsTest,
       InvalidNullPointerSpecConstants) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urProgramSetSpecializationConstants(program, 1, nullptr));
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidSizeCount) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urProgramSetSpecializationConstants(program, 0, &info));
}
