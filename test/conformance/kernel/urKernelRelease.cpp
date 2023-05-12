// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelReleaseTest = uur::urKernelTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelReleaseTest);

TEST_P(urKernelReleaseTest, Success) {
    ASSERT_SUCCESS(urKernelRetain(kernel));
    ASSERT_SUCCESS(urKernelRelease(kernel));
}

TEST_P(urKernelReleaseTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelRelease(nullptr));
}
