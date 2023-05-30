// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelGetInfoTest = uur::urKernelTestWithParam<ur_kernel_info_t>;

UUR_TEST_SUITE_P(
    urKernelGetInfoTest,
    ::testing::Values(UR_KERNEL_INFO_FUNCTION_NAME, UR_KERNEL_INFO_NUM_ARGS,
                      UR_KERNEL_INFO_REFERENCE_COUNT, UR_KERNEL_INFO_CONTEXT,
                      UR_KERNEL_INFO_PROGRAM, UR_KERNEL_INFO_ATTRIBUTES,
                      UR_KERNEL_INFO_NUM_REGS),
    uur::deviceTestWithParamPrinter<ur_kernel_info_t>);

TEST_P(urKernelGetInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS(
        urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));
    property_value.resize(property_size);
    ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                   property_value.data(), nullptr));
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
                     urKernelGetInfo(kernel, UR_KERNEL_INFO_FORCE_UINT32, 0,
                                     nullptr, &bad_enum_length));
}

TEST_P(urKernelGetInfoTest, InvalidSizeZero) {
    size_t n_args = 0;
    ASSERT_EQ_RESULT(
        urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, &n_args, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urKernelGetInfoTest, InvalidSizeSmall) {
    size_t n_args = 0;
    ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                     sizeof(n_args) - 1, &n_args, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urKernelGetInfoTest, InvalidNullPointerPropValue) {
    size_t n_args = 0;
    ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                     sizeof(n_args), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urKernelGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}
