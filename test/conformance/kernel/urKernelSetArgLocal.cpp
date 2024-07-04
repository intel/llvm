// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelSetArgLocalTest : uur::urKernelTest {
    void SetUp() {
        program_name = "mean";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }
    size_t local_mem_size = 4 * sizeof(uint32_t);
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgLocalTest);

TEST_P(urKernelSetArgLocalTest, Success) {
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, 1, local_mem_size, nullptr));
}

TEST_P(urKernelSetArgLocalTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgLocal(nullptr, 1, local_mem_size, nullptr));
}

TEST_P(urKernelSetArgLocalTest, InvalidKernelArgumentIndex) {
    uint32_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
                     urKernelSetArgLocal(kernel, num_kernel_args + 1,
                                         local_mem_size, nullptr));
}
