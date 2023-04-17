// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelGetGroupInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetGroupInfoTest);

TEST_P(urKernelGetGroupInfoTest, Success) {
    size_t work_group_size = 0;
    ASSERT_SUCCESS(urKernelGetGroupInfo(
        kernel, device, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
        sizeof(work_group_size), &work_group_size, nullptr));
    ASSERT_NE(work_group_size, 0);
}

TEST_P(urKernelGetGroupInfoTest, InvalidNullHandleKernel) {
    size_t work_group_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelGetGroupInfo(
                         nullptr, device, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
                         sizeof(work_group_size), &work_group_size, nullptr));
}

TEST_P(urKernelGetGroupInfoTest, InvalidNullHandleDevice) {
    size_t work_group_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelGetGroupInfo(
                         kernel, nullptr, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
                         sizeof(work_group_size), &work_group_size, nullptr));
}

TEST_P(urKernelGetGroupInfoTest, InvalidEnumeration) {
    size_t bad_enum_length = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urKernelGetGroupInfo(nullptr, device,
                                          UR_KERNEL_GROUP_INFO_FORCE_UINT32, 0,
                                          nullptr, &bad_enum_length));
}
