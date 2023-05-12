// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelRetainTest = uur::urKernelTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelRetainTest);

TEST_P(urKernelRetainTest, Success) {
    ASSERT_SUCCESS(urKernelRetain(kernel));
    EXPECT_SUCCESS(urKernelRelease(kernel));
}

TEST_P(urKernelRetainTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelRetain(nullptr));
}
