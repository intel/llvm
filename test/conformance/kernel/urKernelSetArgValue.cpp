// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/known_failure.h"
#include <uur/fixtures.h>

struct urKernelSetArgValueTest : uur::urKernelTest {
    void SetUp() {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }

    uint32_t arg_value = 42;
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgValueTest);

TEST_P(urKernelSetArgValueTest, Success) {
    ASSERT_SUCCESS(
        urKernelSetArgValue(kernel, 2, sizeof(arg_value), nullptr, &arg_value));
}

TEST_P(urKernelSetArgValueTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgValue(nullptr, 2, sizeof(arg_value), nullptr,
                                         &arg_value));
}

TEST_P(urKernelSetArgValueTest, InvalidNullPointerArgValue) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urKernelSetArgValue(kernel, 2, sizeof(arg_value), nullptr, nullptr));
}

TEST_P(urKernelSetArgValueTest, InvalidKernelArgumentIndex) {
    uint32_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
                     urKernelSetArgValue(kernel, num_kernel_args + 1,
                                         sizeof(arg_value), nullptr,
                                         &arg_value));
}

TEST_P(urKernelSetArgValueTest, InvalidKernelArgumentSize) {
    UUR_KNOWN_FAILURE_ON(uur::OpenCL{"Intel(R) UHD Graphics 770"});
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE,
                     urKernelSetArgValue(kernel, 2, 0, nullptr, &arg_value));
}
