// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelGetNativeHandleTest = uur::urKernelTest;
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelGetNativeHandleTest);

TEST_P(urKernelGetNativeHandleTest, Success) {
    ur_native_handle_t native_kernel_handle = nullptr;
    ASSERT_SUCCESS(urKernelGetNativeHandle(kernel, &native_kernel_handle));

    ur_kernel_handle_t native_kernel = nullptr;

    ur_kernel_native_properties_t properties = {
        UR_STRUCTURE_TYPE_KERNEL_NATIVE_PROPERTIES, /*sType*/
        nullptr,                                    /*pNext*/
        true                                        /*isNativeHandleOwned*/
    };
    ASSERT_SUCCESS(urKernelCreateWithNativeHandle(
        native_kernel_handle, context, program, &properties, &native_kernel));

    uint32_t ref_count = 0;
    ASSERT_SUCCESS(urKernelGetInfo(native_kernel,
                                   UR_KERNEL_INFO_REFERENCE_COUNT,
                                   sizeof(ref_count), &ref_count, nullptr));
    ASSERT_NE(ref_count, 0);

    ASSERT_SUCCESS(urKernelRelease(native_kernel));
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
