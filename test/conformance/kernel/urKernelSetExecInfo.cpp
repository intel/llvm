// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelSetExecInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetExecInfoTest);

TEST_P(urKernelSetExecInfoTest, Success) {
    bool property_value = false;
    ASSERT_SUCCESS(
        urKernelSetExecInfo(kernel, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,
                            sizeof(property_value), &property_value));
}

TEST_P(urKernelSetExecInfoTest, InvalidNullHandleKernel) {
    bool property_value = false;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urKernelSetExecInfo(nullptr, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,
                            sizeof(property_value), &property_value));
}

TEST_P(urKernelSetExecInfoTest, InvalidEnumeration) {
    bool property_value = false;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urKernelSetExecInfo(nullptr, UR_KERNEL_EXEC_INFO_FORCE_UINT32,
                            sizeof(property_value), &property_value));
}

TEST_P(urKernelSetExecInfoTest, InvalidNullPointerPropValue) {
    bool property_value = false;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urKernelSetExecInfo(nullptr, UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS,
                            sizeof(property_value), nullptr));
}
