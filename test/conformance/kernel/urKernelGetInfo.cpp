// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelGetInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetInfoTest);

TEST_P(urKernelGetInfoTest, Success) {
    size_t kernel_name_length = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_FUNCTION_NAME, 0,
                                   nullptr, &kernel_name_length));
    std::string queried_kernel_name;
    queried_kernel_name.resize(kernel_name_length);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_FUNCTION_NAME,
                                   queried_kernel_name.size(),
                                   queried_kernel_name.data(), nullptr));
    ASSERT_EQ(queried_kernel_name, kernel_name);
}

TEST_P(urKernelGetInfoTest, InvalidNullHandleKernel) {
    size_t kernel_name_length = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelGetInfo(nullptr, UR_KERNEL_INFO_FUNCTION_NAME, 0,
                                     nullptr, &kernel_name_length));
}

TEST_P(urKernelGetInfoTest, InvalidEnumeration) {
    size_t bad_enum_length = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urKernelGetInfo(nullptr, UR_KERNEL_INFO_FORCE_UINT32, 0,
                                     nullptr, &bad_enum_length));
}
