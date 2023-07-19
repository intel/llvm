// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urKernelGetNativeHandleTest = uur::urKernelTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelGetNativeHandleTest);

TEST_P(urKernelGetNativeHandleTest, Success) {
    ur_native_handle_t native_kernel_handle = nullptr;
    if (auto error = urKernelGetNativeHandle(kernel, &native_kernel_handle)) {
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, error);
    }
}

TEST_P(urKernelGetNativeHandleTest, InvalidNullHandleKernel) {
    ur_native_handle_t native_kernel_handle = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelGetNativeHandle(nullptr, &native_kernel_handle));
}

TEST_P(urKernelGetNativeHandleTest, InvalidNullPointerNativeKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urKernelGetNativeHandle(kernel, nullptr));
}
